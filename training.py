import os
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Параметры
DATA_DIR = 'DataSet'  # путь к датасету
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-3
IMAGE_SIZE = (512, 512)  # фиксированный квадратный размер для всех изображений

# Класс Dataset для загрузки изображений и масок
class BubbleDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_paths = sorted(glob(os.path.join(images_dir, '*')))
        self.masks_paths = sorted(glob(os.path.join(masks_dir, '*')))
        
        # Проверка на соответствие количества изображений и масок
        assert len(self.images_paths) == len(self.masks_paths), "Количество изображений и масок не совпадает!"
        
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # Чтение изображения и маски
        image = np.array(Image.open(self.images_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks_paths[idx]).convert("L"))

        # Приведение маски к бинарному виду (если требуется)
        mask = (mask > 127).astype(np.float32)

        # Применение аугментаций/трансформаций
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # [C, H, W] после ToTensorV2
            mask = augmented['mask']    # [H, W] после ToTensorV2
            
            # Для маски нужно добавить канальную размерность, чтобы получить [1, H, W]
            mask = mask.unsqueeze(0)
        else:
            # Преобразование в тензоры по умолчанию
            image = T.ToTensor()(image)  # [C, H, W]
            mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        return image, mask

# Аугментации с использованием albumentations
def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.LongestMaxSize(max_size=max(IMAGE_SIZE)),  # Изменение размера с сохранением соотношения сторон
            A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1], 
                          border_mode=0),  # Добавление padding до нужного размера
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=max(IMAGE_SIZE)),  # Изменение размера с сохранением соотношения сторон
            A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1], 
                          border_mode=0),  # Добавление padding до нужного размера
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0

            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            # Сохраняем лучшую модель на валидации
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), "best_model.pth")
                print("Saved best model")
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Пути к данным
    train_images_dir = os.path.join(DATA_DIR, 'train/train_images')
    train_masks_dir = os.path.join(DATA_DIR, 'train/train_masks')
    val_images_dir = os.path.join(DATA_DIR, 'valid/valid_images')
    val_masks_dir = os.path.join(DATA_DIR, 'valid/valid_masks')

    # Создание датасетов и загрузчиков данных
    train_dataset = BubbleDataset(train_images_dir, train_masks_dir, transform=get_transforms('train'))
    val_dataset = BubbleDataset(val_images_dir, val_masks_dir, transform=get_transforms('val'))

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    # Создаем модель U-Net (используется предобученный энкодер, можно настроить под задачу)
    model = smp.Unet(encoder_name="resnet34", 
                     encoder_weights="imagenet", 
                     in_channels=3, 
                     classes=1, 
                     activation=None)
    model.to(device)

    # Определяем функцию потерь и оптимизатор
    criterion = nn.BCEWithLogitsLoss()  # для бинарной сегментации
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Запуск обучения
    model = train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS, device)
    print("Training complete.")

if __name__ == '__main__':
    main()
