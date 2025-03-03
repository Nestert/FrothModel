import os
import numpy as np
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Функция для вычисления метрики IoU
def iou_score(pred: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Вычисляет Intersection over Union (IoU) для бинарных масок.
    
    :param pred: Предсказанная карта вероятностей (H x W)
    :param mask: Истинная маска (H x W)
    :param threshold: Порог для бинаризации значений
    :return: Значение IoU
    """
    pred_bin = (pred > threshold).astype(np.uint8)
    mask_bin = (mask > threshold).astype(np.uint8)
    intersection = np.sum(pred_bin * mask_bin)
    union = np.sum(pred_bin) + np.sum(mask_bin) - intersection
    if union == 0:
        return 0.0
    return intersection / union

# Класс Dataset для тестового набора
class BubbleDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_paths = sorted(glob(os.path.join(images_dir, '*')))
        self.masks_paths = sorted(glob(os.path.join(masks_dir, '*')))
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks_paths[idx]).convert("L"))
        mask = (mask > 127).astype(np.float32)  # Бинаризация маски
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = ToTensorV2()(image=image)['image']
            mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask

# Трансформации для тестовых данных (должны соответствовать тем, что использовались при обучении)
def get_transforms():
    return A.Compose([
        A.Resize(800, 608),  # Изменено с 600 на 608, чтобы было кратно 32
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# Функция для оценки модели на тестовом наборе с вычислением IoU
def evaluate_model(model, dataloader, device):
    model.eval()
    total_iou = 0.0
    count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            # Получаем предсказания модели (логиты)
            outputs = model(images)
            # Применяем сигмоиду для получения вероятностей
            outputs = torch.sigmoid(outputs)
            outputs_np = outputs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            # Вычисляем IoU для каждого изображения в батче
            for i in range(outputs_np.shape[0]):
                iou = iou_score(outputs_np[i, 0], masks_np[i, 0])
                total_iou += iou
                count += 1
    average_iou = total_iou / count if count > 0 else 0
    print(f"Среднее значение IoU: {average_iou:.4f}")
    return average_iou

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Задайте пути к тестовым данным
    DATA_DIR = 'DataSet'
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test/test_images')
    TEST_MASKS_DIR = os.path.join(DATA_DIR, 'test/test_masks')
    
    # Создание тестового датасета и загрузчика данных
    test_dataset = BubbleDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Создаем модель U-Net. Если модель уже обучена, то веса будут загружены из файла.
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # При загрузке обученных весов этот параметр не используется
        in_channels=3,
        classes=1,
        activation=None
    )
    
    # Загружаем сохраненные веса обученной модели
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    
    # Оценка модели на тестовом наборе по метрике IoU
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
