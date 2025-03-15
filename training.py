import os
import copy
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --------------------------------------------------
# 1. ПАРАМЕТРЫ
# --------------------------------------------------
DATA_DIR = 'DataSet'  # Корневая папка с данными
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train/train_images')
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, 'train/train_masks')
VAL_IMAGES_DIR = os.path.join(DATA_DIR, 'valid/valid_images')
VAL_MASKS_DIR = os.path.join(DATA_DIR, 'valid/valid_masks')
MODELS_DIR = 'models'  # Папка для сохранения моделей

# Создаем директорию для моделей, если она не существует
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE = 8         # Размер батча
NUM_EPOCHS = 50        # Количество эпох
LR = 1e-3              # Начальный learning rate
IMAGE_SIZE = (512, 512)  # Размер изображений

# Определяем размеры для teacher и student модели
TEACHER_BACKBONE = "timm-mobilenetv3_large_100"
STUDENT_BACKBONE = "timm-mobilenetv3_small_100"
ENCODER_WEIGHTS = "imagenet"

# --------------------------------------------------
# 2. ДАТАСЕТ
# --------------------------------------------------
class BubbleDataset(Dataset):
    """
    Датасет для изображений и бинарных масок.
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(images_dir, '*')))
        self.mask_paths = sorted(glob(os.path.join(masks_dir, '*')))
        assert len(self.image_paths) == len(self.mask_paths), "Число изображений и масок не совпадает!"
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype(np.float32)  # бинаризация

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']    # [C, H, W]
            mask = augmented['mask']      # [H, W]
            mask = mask.unsqueeze(0)      # [1, H, W]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

# --------------------------------------------------
# 3. АУГМЕНТАЦИИ
# --------------------------------------------------
def get_transforms(phase='train'):
    if phase == 'train':
        return A.Compose([
            A.LongestMaxSize(max_size=max(IMAGE_SIZE)),
            A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1], border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(p=0.2, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
                A.MedianBlur(p=1)
            ], p=0.2),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1),
                A.GridDistortion(p=1),
                A.OpticalDistortion(distort_limit=1, p=1)
            ], p=0.2),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.HueSaturationValue(p=1)
            ], p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=max(IMAGE_SIZE)),
            A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1], border_mode=0),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# --------------------------------------------------
# 4. ФУНКЦИЯ ПОТЕРЬ (BCE + Dice)
# --------------------------------------------------
class BCEDiceLoss(nn.Module):
    """
    Комбинированная функция потерь: BCEWithLogitsLoss + Dice Loss.
    """
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        smooth = 1.0
        iflat = inputs.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        dice_loss = 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        return loss

# --------------------------------------------------
# 5. МЕТРИКА IoU
# --------------------------------------------------
def iou_score(outputs, targets, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).float()
    intersection = (outputs * targets).sum(dim=(1,2,3))
    union = (outputs + targets).sum(dim=(1,2,3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean()

# --------------------------------------------------
# 6. ОБУЧЕНИЕ TEACHER МОДЕЛИ
# --------------------------------------------------
def train_model_teacher(model, dataloaders, criterion, optimizer, num_epochs, device):
    best_iou = 0.0
    patience = 10
    patience_counter = 0
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=num_epochs,
        steps_per_epoch=len(dataloaders['train']),
        pct_start=0.3
    )
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_iou = 0.0
            num_samples = 0
            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    batch_iou = iou_score(outputs, masks)
                    running_loss += loss.item() * inputs.size(0)
                    running_iou += batch_iou.item() * inputs.size(0)
                    num_samples += inputs.size(0)
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
            epoch_loss = running_loss / num_samples
            epoch_iou = running_iou / num_samples
            print(f"{phase} Loss: {epoch_loss:.4f} | IoU: {epoch_iou:.4f}")
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            elif phase == 'val':
                patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    model.load_state_dict(best_model_wts)
    return model

# --------------------------------------------------
# 7. ОБУЧЕНИЕ STUDENT МОДЕЛИ С ДИСТИЛЛЯЦИЕЙ
# --------------------------------------------------
def train_model_student(student_model, teacher_model, dataloaders, criterion, optimizer, num_epochs, device, alpha=0.5, temperature=2.0):
    # Teacher модель замораживается
    teacher_model.eval()
    best_iou = 0.0
    patience = 10
    patience_counter = 0
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=num_epochs,
        steps_per_epoch=len(dataloaders['train']),
        pct_start=0.3
    )
    best_model_wts = copy.deepcopy(student_model.state_dict())
    for epoch in range(num_epochs):
        print(f"\nStudent Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                student_model.train()
            else:
                student_model.eval()
            running_loss = 0.0
            running_iou = 0.0
            num_samples = 0
            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    student_outputs = student_model(inputs)
                    base_loss = criterion(student_outputs, masks)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs)
                    # Дистилляционная потеря (с температурным scaling)
                    student_soft = torch.sigmoid(student_outputs / temperature)
                    teacher_soft = torch.sigmoid(teacher_outputs / temperature)
                    distill_loss = F.mse_loss(student_soft, teacher_soft)
                    loss = alpha * base_loss + (1 - alpha) * distill_loss
                    batch_iou = iou_score(student_outputs, masks)
                    running_loss += loss.item() * inputs.size(0)
                    running_iou += batch_iou.item() * inputs.size(0)
                    num_samples += inputs.size(0)
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
            epoch_loss = running_loss / num_samples
            epoch_iou = running_iou / num_samples
            print(f"{phase} Loss: {epoch_loss:.4f} | IoU: {epoch_iou:.4f}")
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(student_model.state_dict())
                patience_counter = 0
            elif phase == 'val':
                patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered for student training")
            break
    student_model.load_state_dict(best_model_wts)
    return student_model

# --------------------------------------------------
# 8. OPTIMIZATION: PRUNING И КВАНТОВАНИЕ
# --------------------------------------------------
def apply_pruning(model, amount=0.3):
    """
    Применяет L1 unstructured pruning к свёрточным слоям.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=amount)
    # Удаляем reparameterization после pruning для уменьшения размера
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            torch.nn.utils.prune.remove(module, 'weight')
    return model

def apply_dynamic_quantization(model):
    """
    Динамическое квантование (работает в основном для Linear слоёв).
    Здесь используется для демонстрации, хотя эффект на свёрточных слоях может быть незначителен.
    """
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return quantized_model

# --------------------------------------------------
# 9. ФУНКЦИИ ИНФЕРЕНСА И ЭКСПОРТА
# --------------------------------------------------
def optimize_model(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model_path = os.path.join(MODELS_DIR, "optimized_model.pt")
    torch.jit.save(traced_model, traced_model_path)
    print(f"Model optimized and saved as '{traced_model_path}'")
    return traced_model

def predict_image(model, image_path, device, transform=None, threshold=0.5):
    model.eval()
    image = np.array(Image.open(image_path).convert("RGB"))
    if transform is not None:
        augmented = transform(image=image)
        input_tensor = augmented['image'].unsqueeze(0).to(device)
    else:
        input_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
        input_tensor = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        pred_mask = (output > threshold).float()
    return pred_mask.squeeze().cpu().numpy()

def convert_to_onnx(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)
    onnx_path = os.path.join(MODELS_DIR, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
    print(f"Model converted and saved as '{onnx_path}'")
    return onnx_path

# --------------------------------------------------
# 10. MAIN: ОБУЧЕНИЕ, ОПТИМИЗАЦИЯ, ДИСТИЛЛЯЦИЯ И ИНФЕРЕНС
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Создаём датасеты и DataLoader'ы
    train_dataset = BubbleDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, transform=get_transforms('train'))
    val_dataset = BubbleDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, transform=get_transforms('val'))
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }

    # 1. Обучение teacher модели (большая модель)
    print("\nОбучение teacher модели...")
    teacher_model = smp.Unet(
        encoder_name=TEACHER_BACKBONE,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    criterion = BCEDiceLoss()
    optimizer_teacher = optim.AdamW(teacher_model.parameters(), lr=LR, weight_decay=1e-4)
    teacher_model = train_model_teacher(teacher_model, dataloaders, criterion, optimizer_teacher, NUM_EPOCHS, device)
    torch.save(teacher_model.state_dict(), os.path.join(MODELS_DIR, "teacher_model.pth"))
    print("Teacher модель сохранена.")

    # Применяем pruning к teacher модели
    print("\nПрименение pruning к teacher модели...")
    teacher_model = apply_pruning(teacher_model, amount=0.3)
    torch.save(teacher_model.state_dict(), os.path.join(MODELS_DIR, "teacher_model_pruned.pth"))
    print("Teacher модель после pruning сохранена.")

    # Применяем динамическое квантование (демонстрация)
    print("\nПрименение динамической квантования к teacher модели...")
    teacher_model_quantized = apply_dynamic_quantization(teacher_model)
    torch.save(teacher_model_quantized.state_dict(), os.path.join(MODELS_DIR, "teacher_model_quantized.pth"))
    print("Teacher модель после квантования сохранена.")

    # 2. Обучение student модели с дистилляцией
    print("\nОбучение student модели с дистилляцией...")
    student_model = smp.Unet(
        encoder_name=STUDENT_BACKBONE,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    optimizer_student = optim.AdamW(student_model.parameters(), lr=LR, weight_decay=1e-4)
    # Замораживаем teacher модель для дистилляции
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    student_model = train_model_student(student_model, teacher_model, dataloaders, criterion, optimizer_student, NUM_EPOCHS, device, alpha=0.5, temperature=2.0)
    torch.save(student_model.state_dict(), os.path.join(MODELS_DIR, "student_model.pth"))
    print("Student модель сохранена.")

    # Оптимизация модели для инференса (TorchScript)
    optimized_model = optimize_model(student_model, device)

    # Конвертация модели в формат ONNX
    convert_to_onnx(student_model, device)

    # Пример инференса на одном изображении
    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        pred_mask = predict_image(student_model, test_image_path, device, transform=get_transforms('val'))
        print("Предсказанная маска имеет форму:", pred_mask.shape)
    else:
        print("test_image.jpg не найден, пропускаем инференс.")

if __name__ == '__main__':
    main()