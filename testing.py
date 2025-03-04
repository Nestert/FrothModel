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
    
    :param pred: Предсказанная маска (H x W)
    :param mask: Истинная маска (H x W)
    :param threshold: Порог для бинаризации значений
    :return: Значение IoU
    """
    # Преобразуем маски в бинарный формат, если они еще не в нем
    pred_bin = (pred > threshold).astype(np.uint8)
    mask_bin = (mask > threshold).astype(np.uint8)
    
    # Добавляем очень малое число, чтобы избежать деления на ноль
    intersection = np.sum(pred_bin * mask_bin)
    union = np.sum(pred_bin) + np.sum(mask_bin) - intersection
    
    if union == 0:
        # Если оба массива пустые, считаем IoU как 1.0
        return 1.0
    
    return intersection / union

# Класс Dataset для тестового набора
class BubbleDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_paths = sorted(glob(os.path.join(images_dir, '*')))
        self.masks_paths = sorted(glob(os.path.join(masks_dir, '*')))
        self.transform = transform
        print(f"Загружено {len(self.images_paths)} изображений и {len(self.masks_paths)} масок")

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # Загружаем изображение
        image_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]
        
        # Проверяем, что файлы существуют
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"ОШИБКА: Файлы не найдены: {image_path} или {mask_path}")
            
        # Загружаем изображение и маску
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Проверяем, содержит ли маска хоть какие-то ненулевые значения
        if idx < 5:  # Только для первых 5 примеров
            non_zero = np.count_nonzero(mask)
            print(f"Индекс {idx}, загруженная маска: ненулевых пикселей = {non_zero}")
            
        # Бинаризация маски
        mask = (mask > 127).astype(np.float32)
        
        if idx < 5:
            print(f"Индекс {idx}, после бинаризации: ненулевых пикселей = {np.count_nonzero(mask)}")
        
        if self.transform:
            # Применяем трансформации
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Убедимся, что маска - это тензор с правильным числом измерений
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)
            
            # Проверяем размерность маски и добавляем канал, если нужно
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # Добавляем канал [1, H, W]
                
            if idx < 5:
                print(f"Индекс {idx}, после трансформации: форма={mask.shape}, ненулевых={torch.count_nonzero(mask).item()}")
        else:
            image = ToTensorV2()(image=image)['image']
            mask = torch.from_numpy(mask).unsqueeze(0)
            
        return image, mask

# Трансформации для тестовых данных (должны соответствовать тем, что использовались при обучении)
def get_transforms():
    IMAGE_SIZE = (512, 512)  # Такой же размер, как и при обучении
    return A.Compose([
        A.LongestMaxSize(max_size=max(IMAGE_SIZE)),
        A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1],
                      border_mode=0),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], p=1.0)

# Функция для оценки модели на тестовом наборе с вычислением IoU
def evaluate_model(model, dataloader, device):
    model.eval()
    total_iou = 0.0
    count = 0
    batch_count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            batch_count += 1
            if batch_count <= 2:  # Только для первых двух батчей
                print(f"\nБатч {batch_count}, форма масок: {masks.shape}")
                print(f"Сумма масок в батче: {masks.sum().item()}")
                print(f"Мин/Макс значения масок: {masks.min().item()}/{masks.max().item()}")
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Получаем предсказания модели (логиты)
            outputs = model(images)
            
            # Применяем сигмоиду для получения вероятностей
            outputs = torch.sigmoid(outputs)
            
            # Применяем пороговое значение для бинаризации
            thresholded_outputs = (outputs > 0.5).float()
            
            # Переводим тензоры в numpy для вычисления IoU
            thresholded_outputs_np = thresholded_outputs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # Вычисляем IoU для каждого изображения в батче
            for i in range(thresholded_outputs_np.shape[0]):
                iou = iou_score(thresholded_outputs_np[i, 0], masks_np[i, 0])
                total_iou += iou
                count += 1
                
            # Добавляем отладочный вывод для первых нескольких предсказаний
            if batch_count <= 2:
                for i in range(min(thresholded_outputs_np.shape[0], 2)):
                    print(f"Пример {i+1}: IoU = {iou_score(thresholded_outputs_np[i, 0], masks_np[i, 0]):.4f}")
                    print(f"   Сумма предсказания: {thresholded_outputs_np[i, 0].sum()}, "
                          f"сумма маски: {masks_np[i, 0].sum()}")
    
    average_iou = total_iou / count if count > 0 else 0
    print(f"Среднее значение IoU: {average_iou:.4f}")
    return average_iou

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Задаем пути к тестовым данным
    DATA_DIR = 'DataSet'
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test/test_images')
    TEST_MASKS_DIR = os.path.join(DATA_DIR, 'test/test_masks')
    
    # Проверка наличия тестовых файлов
    test_images = sorted(glob(os.path.join(TEST_IMAGES_DIR, '*')))
    test_masks = sorted(glob(os.path.join(TEST_MASKS_DIR, '*')))
    print(f"Найдено тестовых изображений: {len(test_images)}")
    print(f"Найдено тестовых масок: {len(test_masks)}")
    
    if len(test_images) == 0 or len(test_masks) == 0:
        print("ОШИБКА: Не найдены тестовые файлы!")
        return
    
    # Создание тестового датасета и загрузчика данных
    test_dataset = BubbleDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    print(f"Создан DataLoader с {len(test_dataset)} тестовыми примерами")
    
    # Создаем модель U-Net
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    
    # Загружаем сохраненные веса обученной модели
    model_path = "best_model.pth"
    if os.path.exists(model_path):
        print(f"Загрузка весов модели из {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        # Оценка модели на тестовом наборе по метрике IoU
        test_iou = evaluate_model(model, test_loader, device)
        print(f"Итоговый Test IoU: {test_iou:.4f}")
    else:
        print(f"ОШИБКА: Файл модели {model_path} не найден!")

if __name__ == '__main__':
    main()
