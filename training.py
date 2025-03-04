import os
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --------------------------------------------------
# 1. ПАРАМЕТРЫ
# --------------------------------------------------
DATA_DIR = 'DataSet'   # Корневая папка с данными
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train/train_images')
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, 'train/train_masks')
VAL_IMAGES_DIR = os.path.join(DATA_DIR, 'valid/valid_images')
VAL_MASKS_DIR = os.path.join(DATA_DIR, 'valid/valid_masks')

# Размер батча, количество эпох, начальный learning rate
BATCH_SIZE = 4
NUM_EPOCHS = 50
LR = 1e-3
IMAGE_SIZE = (512, 512)  # Размер, к которому приводим все изображения

# --------------------------------------------------
# 2. ДАТАСЕТ
# --------------------------------------------------
class BubbleDataset(Dataset):
    """
    Класс датасета для изображений и бинарных масок пузырьков (или других объектов).
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(images_dir, '*')))
        self.mask_paths = sorted(glob(os.path.join(masks_dir, '*')))
        assert len(self.image_paths) == len(self.mask_paths), \
            "Число изображений и масок не совпадает!"

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Загружаем изображение
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        # Загружаем маску
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        # Приводим маску к бинарному виду (0 или 1)
        mask = (mask > 127).astype(np.float32)

        # Применяем аугментации, если они заданы
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # [C, H, W]
            mask = augmented['mask']    # [H, W]
            # Добавляем ось канала для маски -> [1, H, W]
            mask = mask.unsqueeze(0)
        else:
            # Если аугментации не заданы, просто приводим к тензору
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

# --------------------------------------------------
# 3. АУГМЕНТАЦИИ
# --------------------------------------------------
def get_transforms(phase='train'):
    """
    Возвращает Compose из Albumentations для заданной фазы (train или val).
    """
    if phase == 'train':
        return A.Compose([
            # Изменение размера с сохранением соотношения сторон
            A.LongestMaxSize(max_size=max(IMAGE_SIZE)),
            # Дополнение паддингом до нужного размера
            A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1],
                          border_mode=0, value=(0, 0, 0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Дополнительные цветовые аугментации
            A.ColorJitter(p=0.2, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # Нормализация под ImageNet
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=max(IMAGE_SIZE)),
            A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1],
                          border_mode=0, value=(0, 0, 0)),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# --------------------------------------------------
# 4. ФУНКЦИЯ ПОТЕРЬ (BCE + Dice)
# --------------------------------------------------
class BCEDiceLoss(nn.Module):
    """
    Комбинированная функция потерь: BCEWithLogits + Dice Loss.
    """
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # BCE
        bce_loss = self.bce(inputs, targets)

        # Sigmoid для расчёта Dice
        inputs = torch.sigmoid(inputs)
        smooth = 1.0
        # Приведём тензоры к одному размеру
        iflat = inputs.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        dice_loss = 1 - ((2.0 * intersection + smooth) /
                         (iflat.sum() + tflat.sum() + smooth))

        # Итоговая комбинированная ошибка
        loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        return loss

# --------------------------------------------------
# 5. МЕТРИКИ (IoU, Dice)
# --------------------------------------------------
def iou_score(outputs, targets, threshold=0.5):
    """
    Вычисление IoU для бинарной сегментации.
    outputs: [B, 1, H, W]
    targets: [B, 1, H, W]
    threshold: порог для бинаризации предсказаний
    """
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).float()
    intersection = (outputs * targets).sum(dim=(1, 2, 3))
    union = (outputs + targets).sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean()

# --------------------------------------------------
# 6. ОБУЧЕНИЕ МОДЕЛИ
# --------------------------------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    best_iou = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
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

                    # Считаем IoU
                    batch_iou = iou_score(outputs, masks)
                    running_iou += batch_iou.item() * inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    num_samples += inputs.size(0)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            epoch_loss = running_loss / num_samples
            epoch_iou = running_iou / num_samples

            print(f"{phase} Loss: {epoch_loss:.4f} | {phase} IoU: {epoch_iou:.4f}")

            # Сохраняем лучшую модель по метрике IoU на валидации
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                torch.save(model.state_dict(), "best_model.pth")
                print("Model saved (best IoU so far).")

    print("\nTraining complete.")
    print(f"Best validation IoU: {best_iou:.4f}")
    return model

# --------------------------------------------------
# 7. ИНФЕРЕНС (ПРОГНОЗ)
# --------------------------------------------------
def predict_image(model, image_path, device, transform=None, threshold=0.5):
    """
    Функция для прогноза маски на одном изображении.
    """
    model.eval()
    # Загрузка изображения
    image = np.array(Image.open(image_path).convert("RGB"))

    # Применяем трансформации
    if transform is not None:
        augmented = transform(image=image)
        input_tensor = augmented['image'].unsqueeze(0).to(device)
    else:
        # Без аугментаций
        input_tensor = torch.from_numpy(image).permute(2, 0, 1).float()/255.
        input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        pred_mask = (output > threshold).float()
    # Приведём маску к numpy
    pred_mask = pred_mask.squeeze().cpu().numpy()  # shape: [H, W]
    return pred_mask

# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Создаём датасеты
    train_dataset = BubbleDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, transform=get_transforms('train'))
    val_dataset = BubbleDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, transform=get_transforms('val'))

    # Создаём DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }

    # Определяем модель (Unet + ResNet34)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None  # будем использовать сигмоиду внутри кода
    ).to(device)

    # Функция потерь (комбинированная BCE + Dice)
    criterion = BCEDiceLoss(bce_weight=0.5)

    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Обучаем модель
    model = train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS, device)

    # Пример инференса на одном изображении
    # (путь к любому изображению, где нужно спрогнозировать маску)
    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        pred_mask = predict_image(model, test_image_path, device, transform=get_transforms('val'))
        print("Предсказанная маска имеет форму:", pred_mask.shape)
    else:
        print("test_image.jpg не найден, пропускаем инференс.")

if __name__ == '__main__':
    main()
