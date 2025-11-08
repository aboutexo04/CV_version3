"""
Vision Transformer Base 384 - Optimized for Document Classification
주요 개선사항:
1. TTA (Test Time Augmentation) 추가 → 안정적인 예측
2. Focal Loss → 클래스 불균형 직접 대응
3. Dropout 조정 → 과적합 방지
4. Better validation strategy
5. Ensemble with confidence weighting
"""

import os
import time
import random
import pickle
import cv2
import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

# ==================== Reproducibility ====================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

print("="*80)
print("🚀 VISION TRANSFORMER BASE 384 - OPTIMIZED")
print("="*80)
print("\nKey Improvements:")
print("  ✓ Test Time Augmentation (TTA)")
print("  ✓ Focal Loss for class imbalance")
print("  ✓ Confidence-weighted ensemble")
print("  ✓ Better regularization")
print("  ✓ Optimized training schedule")
print("="*80)

# ==================== Configuration ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[CONFIG] Device: {device}")

# Model parameters
model_name = 'vit_base_patch16_384'
img_size = 384
LR = 2e-5  # 더 보수적으로
EPOCHS = 35  # 약간 줄임
BATCH_SIZE = 8
num_workers = 4
n_splits = 5
warmup_epochs = 3  # warmup 늘림
gradient_accumulation_steps = 2

# TTA settings
TTA_TRANSFORMS = 5  # TTA 횟수

# Paths
train_csv = './data/train.csv'
test_csv = './data/sample_submission.csv'
train_folder = './data/train/'
test_folder = './data/test/'
model_save_path = './models_vit_base_384_opt/'

os.makedirs(model_save_path, exist_ok=True)

print(f"[CONFIG] Model: {model_name}")
print(f"[CONFIG] Learning rate: {LR}")
print(f"[CONFIG] TTA transforms: {TTA_TRANSFORMS}")

# ==================== Focal Loss ====================
class FocalLoss(nn.Module):
    """클래스 불균형에 강한 Focal Loss"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==================== Mixup/CutMix ====================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    """Apply mixup or cutmix - 더 보수적으로"""
    if alpha <= 0 or np.random.rand() > 0.7:  # 70%만 적용
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    if np.random.rand() < cutmix_prob:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    else:
        x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================== Dataset Class ====================
class ImageDataset(Dataset):
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, int(target)

# ==================== Training Functions ====================
def train_one_epoch(loader, model, optimizer, loss_fn, device, epoch, scaler, use_mixup=True):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
    for batch_idx, (image, targets) in enumerate(pbar):
        image = image.to(device)
        targets = targets.to(device)

        if use_mixup:
            mixed_image, targets_a, targets_b, lam = mixup_cutmix_data(
                image, targets, alpha=0.3, cutmix_prob=0.5  # alpha 줄임
            )
        else:
            mixed_image, targets_a, targets_b, lam = image, targets, targets, 1.0

        with torch.cuda.amp.autocast():
            preds = model(mixed_image)
            loss = mixed_criterion(loss_fn, preds, targets_a, targets_b, lam)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * gradient_accumulation_steps
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    return {"train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1}

def validate(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for image, targets in pbar:
            image = image.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast():
                preds = model(image)
                loss = loss_fn(preds, targets)

            val_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())

    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')

    return {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}, val_f1

# ==================== TTA 함수 ====================
def predict_with_tta(model, loader, device, n_tta=5):
    """Test Time Augmentation으로 더 안정적인 예측"""
    model.eval()
    
    # TTA transforms
    tta_transforms = [
        val_transform,  # Original
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Rotate(limit=90, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Rotate(limit=-90, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),
    ]
    
    all_tta_preds = []
    
    for tta_idx in range(min(n_tta, len(tta_transforms))):
        # Create new dataset with TTA transform
        tta_dataset = ImageDataset(test_csv, test_folder, transform=tta_transforms[tta_idx])
        tta_loader = DataLoader(tta_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        
        tta_preds = []
        with torch.no_grad():
            for image, _ in tta_loader:
                image = image.to(device)
                with torch.cuda.amp.autocast():
                    preds = model(image)
                    probs = torch.softmax(preds, dim=1)
                tta_preds.append(probs.cpu().numpy())
        
        tta_preds = np.concatenate(tta_preds, axis=0)
        all_tta_preds.append(tta_preds)
    
    # Average TTA predictions
    avg_tta_preds = np.mean(all_tta_preds, axis=0)
    return avg_tta_preds

# ==================== Data Augmentation ====================
print("\n[AUGMENTATION] Setting up transformations...")

# 약간 더 보수적인 augmentation
trn_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
    A.ShiftScaleRotate(
        shift_limit=0.08,
        scale_limit=0.12,
        rotate_limit=25,
        p=0.5,
        border_mode=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    ),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=1.0),
    ], p=0.4),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 25.0), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    ], p=0.25),
    A.CoarseDropout(
        max_holes=6,
        max_height=28,
        max_width=28,
        min_holes=1,
        fill_value=255,
        p=0.25
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ==================== K-Fold Cross Validation ====================
print("\n" + "="*80)
print("STARTING K-FOLD CROSS VALIDATION")
print("="*80)

start_time = time.time()

full_dataset = ImageDataset(train_csv, train_folder, transform=trn_transform)
X = full_dataset.df[:, 0]
y = full_dataset.df[:, 1].astype(int)

print(f"\nDataset size: {len(full_dataset)}")
print(f"Number of classes: 17")

# 클래스 분포 확인
class_counts = pd.Series(y).value_counts().sort_index()
print("\nClass distribution:")
for cls in range(17):
    count = class_counts.get(cls, 0)
    print(f"  Class {cls:2d}: {count:4d} samples")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

all_fold_f1_scores = []
fold_confidences = []  # 각 fold의 confidence 저장

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{n_splits}")
    print(f"{'='*80}")

    train_dataset = ImageDataset(train_csv, train_folder, transform=trn_transform)
    valid_dataset = ImageDataset(train_csv, train_folder, transform=val_transform)

    train_subset = torch.utils.data.Subset(train_dataset, train_index)
    valid_subset = torch.utils.data.Subset(valid_dataset, valid_index)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train samples: {len(train_subset)}, Valid samples: {len(valid_subset)}")

    # Create model
    print(f"\nCreating {model_name}...")
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17,
        drop_rate=0.15,  # dropout 약간 증가
        drop_path_rate=0.15,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Focal Loss 사용
    loss_fn = FocalLoss(alpha=1.0, gamma=2.0)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.02)  # weight decay 증가
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler()

    best_val_f1 = 0
    patience = 10  # patience 줄임
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.7f})")

        train_metrics = train_one_epoch(
            train_loader, model, optimizer, loss_fn, device, epoch, scaler,
            use_mixup=(epoch > warmup_epochs)
        )

        val_metrics, val_f1 = validate(valid_loader, model, loss_fn, device)
        scheduler.step()

        print(f"Train - Loss: {train_metrics['train_loss']:.4f}, "
              f"Acc: {train_metrics['train_acc']:.4f}, F1: {train_metrics['train_f1']:.4f}")
        print(f"Valid - Loss: {val_metrics['val_loss']:.4f}, "
              f"Acc: {val_metrics['val_acc']:.4f}, F1: {val_metrics['val_f1']:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            model_filename = os.path.join(model_save_path, f'model_fold{fold}_best.pkl')
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)

            print(f"✓ Best F1 improved: {best_val_f1:.4f} - Model saved")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    all_fold_f1_scores.append(best_val_f1)
    fold_confidences.append(best_val_f1)  # F1을 confidence로 사용
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1} COMPLETE - Best F1: {best_val_f1:.4f}")
    print(f"{'='*80}")

# ==================== Summary ====================
elapsed_time = time.time() - start_time

print("\n" + "="*80)
print("CROSS VALIDATION COMPLETE")
print("="*80)

mean_f1 = np.mean(all_fold_f1_scores)
std_f1 = np.std(all_fold_f1_scores)

print(f"\nResults per fold:")
for i, f1 in enumerate(all_fold_f1_scores):
    print(f"  Fold {i+1}: F1 = {f1:.4f}")

print(f"\nOverall Performance:")
print(f"  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  Training time: {elapsed_time/60:.1f} minutes")

print("\n" + "="*80)
print("GENERATING PREDICTIONS WITH TTA")
print("="*80)

# Confidence weights 계산
confidence_weights = np.array(fold_confidences)
confidence_weights = confidence_weights / confidence_weights.sum()

print(f"\nConfidence weights:")
for i, w in enumerate(confidence_weights):
    print(f"  Fold {i+1}: {w:.4f}")

# TTA를 사용한 앙상블 예측
all_predictions = []

for fold in range(n_splits):
    print(f"\nPredicting with Fold {fold + 1} model (with TTA)...")

    model_file = os.path.join(model_save_path, f'model_fold{fold}_best.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    model.to(device)
    
    # TTA 적용
    fold_preds = predict_with_tta(model, None, device, n_tta=TTA_TRANSFORMS)
    all_predictions.append(fold_preds)

# Confidence-weighted ensemble
print("\nApplying confidence-weighted ensemble...")
weighted_preds = np.zeros_like(all_predictions[0])
for fold_pred, weight in zip(all_predictions, confidence_weights):
    weighted_preds += fold_pred * weight

final_preds = np.argmax(weighted_preds, axis=1)

# Calculate imbalance
pred_counts = pd.Series(final_preds).value_counts().sort_index()
imbalance = pred_counts.max() / pred_counts.min()
print(f"\nPrediction imbalance: {imbalance:.2f}x")
print(f"\nClass distribution:")
for cls in range(17):
    count = pred_counts.get(cls, 0)
    print(f"  Class {cls:2d}: {count:4d} samples")

# Save submission
pred_df = pd.DataFrame(pd.read_csv(test_csv).values, columns=['ID', 'target'])
pred_df['target'] = final_preds

now = datetime.now()
date_str = now.strftime("%Y%m%d")
time_str = now.strftime("%H%M%S")
f1_str = f"{mean_f1:.4f}".replace(".", "")
imb_str = f"{imbalance:.2f}".replace(".", "")

filename = f"submission_{date_str}_{time_str}_vitbase384opt_f1{f1_str}_imb{imb_str}_tta{TTA_TRANSFORMS}.csv"
pred_df.to_csv(f"./data/{filename}", index=False)

print(f"\n✓ Predictions saved to ./data/{filename}")

print("\n" + "="*80)
print("🎉 OPTIMIZED ViT-BASE-384 TRAINING COMPLETE!")
print("="*80)
print(f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Prediction Imbalance: {imbalance:.2f}x")
print(f"TTA Transforms: {TTA_TRANSFORMS}")
print(f"Total Time: {elapsed_time/60:.1f} minutes")
print(f"Models saved to: {model_save_path}")
print(f"Submission file: ./data/{filename}")
print("="*80)