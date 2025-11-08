import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from datetime import datetime
import logging
import sys

# Seed 고정
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# 하이퍼파라미터 설정
CFG = {
    'IMG_SIZE': 384,
    'EPOCHS': 50,
    'BATCH_SIZE': 32,
    'LR': 1e-4,
    'SEED': 42,
    'NUM_CLASSES': 17,
    'N_FOLDS': 5,
    'PATIENCE': 10,  # Early Stopping patience
    'MODEL_NAME': 'convnext_base.fb_in22k_ft_in1k',  # ConvNeXt-Base model
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Dataset 정의 (Albumentations 사용)
class DocumentDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]['ID'])
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            image = self.transform(image=image)['image']

        label = self.df.iloc[idx]['target']
        return image, label

# ViT Winner 증강 (train_vit_base_384_WINNER.py에서 가져옴)
train_transform = A.Compose([
    A.Resize(height=CFG['IMG_SIZE'], width=CFG['IMG_SIZE']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=180, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.15,
        rotate_limit=30,
        p=0.6,
        border_mode=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    ),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    ], p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    ], p=0.3),
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        min_holes=1,
        fill_value=255,
        p=0.3
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=CFG['IMG_SIZE'], width=CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(height=CFG['IMG_SIZE'], width=CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 모델 정의
def get_model(num_classes=CFG['NUM_CLASSES']):
    model = timm.create_model(CFG['MODEL_NAME'], pretrained=True, num_classes=num_classes)
    return model

# 학습 함수
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# 검증 함수
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# 예측 함수
def predict(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Predicting')
        for images in pbar:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())

    return predictions

# 로거 설정
def setup_logger(exp_dir, timestamp):
    log_file = os.path.join(exp_dir, f'train_{timestamp}.log')

    # 로거 생성
    logger = logging.getLogger('DocumentClassification')
    logger.setLevel(logging.INFO)

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 메인 실행
def main():
    # 실험 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    # 로거 설정
    logger = setup_logger(exp_dir, timestamp)

    logger.info(f"=" * 60)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Device: {CFG['DEVICE']}")
    logger.info(f"Model: {CFG['MODEL_NAME']}")
    logger.info(f"=" * 60)
    logger.info(f"Configuration:")
    for key, value in CFG.items():
        logger.info(f"  {key}: {value}")

    # 데이터 로드
    train_df = pd.read_csv('./data/train.csv')
    logger.info(f"Total training samples: {len(train_df)}")

    # 클래스 분포 및 불균형 지수 계산
    logger.info("\n" + "=" * 60)
    logger.info("Class Distribution Analysis")
    logger.info("=" * 60)

    class_counts = train_df['target'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        percentage = (count / len(train_df)) * 100
        logger.info(f"Class {class_id}: {count} samples ({percentage:.2f}%)")

    # Imbalance Ratio: 가장 많은 클래스 / 가장 적은 클래스
    max_samples = class_counts.max()
    min_samples = class_counts.min()
    imbalance_ratio = max_samples / min_samples
    logger.info("-" * 60)
    logger.info(f"Max samples per class: {max_samples}")
    logger.info(f"Min samples per class: {min_samples}")
    logger.info(f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}")

    # Gini Coefficient (불균형 정도를 0~1로 표현, 0: 완벽한 균형, 1: 완전 불균형)
    sorted_counts = np.sort(class_counts.values)
    n = len(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
    logger.info(f"Gini Coefficient: {gini:.4f}")
    logger.info("=" * 60)

    # K-Fold Cross Validation 설정
    skf = StratifiedKFold(n_splits=CFG['N_FOLDS'], shuffle=True, random_state=CFG['SEED'])

    fold_results = []
    oof_predictions = np.zeros(len(train_df))
    test_predictions_per_fold = []

    # 각 Fold 학습
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target']), 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"FOLD {fold}/{CFG['N_FOLDS']}")
        logger.info("=" * 60)

        # Train/Validation 데이터 분할
        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        val_data = train_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

        # Dataset 생성
        train_dataset = DocumentDataset(train_data, './data/train', transform=train_transform)
        val_dataset = DocumentDataset(val_data, './data/train', transform=val_transform)

        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG['BATCH_SIZE'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG['BATCH_SIZE'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 모델, 손실함수, 옵티마이저 설정
        model = get_model().to(CFG['DEVICE'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=CFG['LR'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'], eta_min=1e-6)

        # 학습
        logger.info(f"Starting training for Fold {fold}...")
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(CFG['EPOCHS']):
            logger.info(f"\nFold {fold} - Epoch {epoch+1}/{CFG['EPOCHS']}")
            logger.info("-" * 50)

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CFG['DEVICE'])
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, CFG['DEVICE'])
            scheduler.step()

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"LR: {scheduler.get_last_lr()[0]:.6f}")

            # 최고 성능 모델 저장 및 Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(exp_dir, f'best_model_fold{fold}.pth')
                torch.save(model.state_dict(), model_path)
                logger.info(f"Best model for Fold {fold} saved! (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{CFG['PATIENCE']}")

            # Early Stopping 체크
            if patience_counter >= CFG['PATIENCE']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        logger.info(f"\nFold {fold} training completed! Best validation accuracy: {best_val_acc:.2f}%")

        # Validation 데이터로 F1 Score 계산
        logger.info(f"Calculating F1 score for Fold {fold}...")
        model_path = os.path.join(exp_dir, f'best_model_fold{fold}.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()

        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Fold {fold} - Calculating F1'):
                images = images.to(CFG['DEVICE'])
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.numpy())

        fold_f1 = f1_score(val_labels, val_preds, average='macro')
        logger.info(f"Fold {fold} - Validation F1 Score (macro): {fold_f1:.4f}")

        # OOF 예측 저장
        oof_predictions[val_idx] = val_preds

        # Fold 결과 저장
        fold_results.append({
            'fold': fold,
            'val_acc': best_val_acc,
            'val_f1': fold_f1
        })

        # 테스트 데이터 예측 (각 Fold)
        logger.info(f"Generating test predictions for Fold {fold}...")
        test_files = sorted(os.listdir('./data/test'))

        class TestDataset(Dataset):
            def __init__(self, file_list, img_dir, transform=None):
                self.file_list = file_list
                self.img_dir = img_dir
                self.transform = transform

            def __len__(self):
                return len(self.file_list)

            def __getitem__(self, idx):
                img_path = os.path.join(self.img_dir, self.file_list[idx])
                image = np.array(Image.open(img_path).convert('RGB'))
                if self.transform:
                    image = self.transform(image=image)['image']
                return image

        test_dataset = TestDataset(test_files, './data/test', transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG['BATCH_SIZE'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        fold_test_preds = predict(model, test_loader, CFG['DEVICE'])
        test_predictions_per_fold.append(fold_test_preds)

    # 전체 Fold 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("K-FOLD CROSS VALIDATION RESULTS")
    logger.info("=" * 60)
    for result in fold_results:
        logger.info(f"Fold {result['fold']}: Val Acc = {result['val_acc']:.2f}%, Val F1 = {result['val_f1']:.4f}")

    avg_acc = np.mean([r['val_acc'] for r in fold_results])
    avg_f1 = np.mean([r['val_f1'] for r in fold_results])
    logger.info("-" * 60)
    logger.info(f"Average Val Acc: {avg_acc:.2f}%")
    logger.info(f"Average Val F1: {avg_f1:.4f}")

    # OOF F1 Score
    oof_f1 = f1_score(train_df['target'].values, oof_predictions, average='macro')
    logger.info(f"OOF F1 Score: {oof_f1:.4f}")
    logger.info("=" * 60)

    # 최종 F1은 OOF F1 사용
    f1 = oof_f1

    # 테스트 데이터 앙상블 예측 (모든 Fold의 평균)
    logger.info("\n" + "=" * 60)
    logger.info("Generating final ensemble predictions...")
    test_files = sorted(os.listdir('./data/test'))

    # 모든 fold의 예측을 평균내어 최종 예측 생성
    ensemble_predictions = np.array(test_predictions_per_fold).mean(axis=0).astype(int)
    predictions = ensemble_predictions

    # Submission 파일 생성
    submission = pd.DataFrame({
        'ID': test_files,
        'target': predictions
    })
    submission_filename = f"submission_{timestamp}_f1_{f1:.4f}.csv"
    submission_path = os.path.join(exp_dir, submission_filename)
    submission.to_csv(submission_path, index=False)
    logger.info(f"Submission file saved as '{submission_path}'")
    logger.info("=" * 60)
    logger.info("Experiment completed successfully!")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
