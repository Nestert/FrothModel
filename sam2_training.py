import os
import numpy as np
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Импортируем SAM2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

# --------------------------------------------------
# 1. ПАРАМЕТРЫ
# --------------------------------------------------
DATA_DIR = 'DataSet'  # Корневая папка с данными
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train/train_images')
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, 'train/train_masks')
VAL_IMAGES_DIR = os.path.join(DATA_DIR, 'valid/valid_images')
VAL_MASKS_DIR = os.path.join(DATA_DIR, 'valid/valid_masks')

# Параметры обучения
BATCH_SIZE = 8  # SAM2 требует больше памяти, уменьшаем размер батча
NUM_EPOCHS = 30
LEARNING_RATE = 1e-5  # Уменьшенный LR для тонкой настройки SAM2
IMAGE_SIZE = 1024  # SAM2 обычно работает с большими изображениями

# Параметры SAM2
SAM_CHECKPOINT = "sam2_h.pth"  # Путь к предобученным весам SAM2
SAM_TYPE = "vit_h"  # Тип модели: vit_h (huge), vit_l (large), vit_b (base)

# --------------------------------------------------
# 2. ДАТАСЕТ
# --------------------------------------------------
class BubbleDatasetSAM(Dataset):
    """
    Класс датасета для SAM2 - изображения и бинарные маски пузырьков.
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(images_dir, '*')))
        self.mask_paths = sorted(glob(os.path.join(masks_dir, '*')))
        assert len(self.image_paths) == len(self.mask_paths), \
            "Число изображений и масок не совпадает!"

        self.transform = transform
        self.resize_transform = ResizeLongestSide(IMAGE_SIZE)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Загружаем изображение
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        
        # Загружаем маску
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype(np.float32)
        
        # Применяем resize трансформацию SAM2
        image_resized = self.resize_transform.apply_image(image)
        mask_resized = self.resize_transform.apply_image(mask[:, :, None])[:, :, 0]
        
        # Получаем координаты для подсказок точками (prompt points)
        # Находим центры объектов на маске для подсказок
        from scipy import ndimage
        labeled_mask, num_features = ndimage.label(mask_resized)
        
        prompt_points = []
        prompt_labels = []
        
        # Если есть объекты, получаем их центры как точки подсказки
        if num_features > 0:
            for i in range(1, num_features + 1):
                # Получаем центр объекта
                y, x = ndimage.center_of_mass(labeled_mask == i)
                prompt_points.append([x, y])
                prompt_labels.append(1)  # 1 означает, что это foreground
        
        # Добавляем несколько отрицательных подсказок (точки фона)
        # Находим точки, где определённо фон (где маска равна 0)
        if np.sum(mask_resized == 0) > 0:
            background_y, background_x = np.where(mask_resized == 0)
            # Выбираем до 5 случайных точек фона
            if len(background_y) > 0:
                bg_indices = np.random.choice(len(background_y), min(5, len(background_y)), replace=False)
                for i in bg_indices:
                    prompt_points.append([background_x[i], background_y[i]])
                    prompt_labels.append(0)  # 0 означает, что это background
        
        # Преобразуем в тензоры
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)
        
        # Если есть подсказки, преобразуем их в тензоры
        if prompt_points:
            prompt_points = torch.tensor(prompt_points, dtype=torch.float)
            prompt_labels = torch.tensor(prompt_labels, dtype=torch.int)
        else:
            # Если подсказок нет, создаем пустые тензоры
            prompt_points = torch.zeros((0, 2), dtype=torch.float)
            prompt_labels = torch.zeros(0, dtype=torch.int)
        
        # Дополнительно применяем трансформации, если они заданы
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        # Вычисляем оригинальные размеры изображения (нужно для SAM2)
        original_size = torch.tensor([image.shape[0], image.shape[1]], dtype=torch.int)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'prompt_points': prompt_points,
            'prompt_labels': prompt_labels,
            'original_size': original_size,
            'image_path': self.image_paths[idx]
        }

# --------------------------------------------------
# 3. МОДЕЛЬ SAM2
# --------------------------------------------------
class SAM2Wrapper(nn.Module):
    """
    Обертка для модели SAM2, упрощающая тренировку и инференс.
    """
    def __init__(self, checkpoint_path, model_type):
        super().__init__()
        # Инициализируем модель SAM2
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        # Заморозим часть весов для тонкой настройки
        self._freeze_base_layers()
        
    def _freeze_base_layers(self):
        """
        Замораживаем базовые слои модели, оставляя для обучения только
        последние слои декодера для тонкой настройки под наш датасет.
        """
        # Замораживаем энкодер изображения (он уже хорошо предобучен)
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
            
        # Замораживаем часть prompt энкодера
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
            
        # Оставляем для обучения только последние слои маск-декодера
        for i, block in enumerate(self.sam.mask_decoder.transformer.blocks):
            # Размораживаем только последние 2 блока
            if i >= len(self.sam.mask_decoder.transformer.blocks) - 2:
                for param in block.parameters():
                    param.requires_grad = True
            else:
                for param in block.parameters():
                    param.requires_grad = False
    
    def forward(self, images, prompt_points=None, prompt_labels=None, original_sizes=None):
        """
        Forward pass для SAM2.
        
        Args:
            images: [B, 3, H, W] - батч изображений
            prompt_points: [B, N, 2] - координаты точек-подсказок
            prompt_labels: [B, N] - метки точек (1 - объект, 0 - фон)
            original_sizes: [B, 2] - оригинальные размеры изображений
            
        Returns:
            masks: предсказанные маски
        """
        B = images.shape[0]
        
        # Предобрабатываем изображения через image encoder
        image_embeddings = self.sam.image_encoder(images)
        
        # Подготавливаем sparse embeddings для prompt точек
        sparse_embeddings = None
        dense_embeddings = None
        
        # Если есть prompt points, используем их
        if prompt_points is not None and prompt_labels is not None:
            batch_sparse_embeddings = []
            
            for i in range(B):
                # Получаем точки и метки для текущего изображения
                points = prompt_points[i]
                labels = prompt_labels[i]
                
                if points.numel() > 0:
                    # Кодируем точки через prompt encoder
                    sparse_emb, dense_emb = self.sam.prompt_encoder(
                        points=points.unsqueeze(0),
                        labels=labels.unsqueeze(0),
                        boxes=None,
                        masks=None
                    )
                    batch_sparse_embeddings.append(sparse_emb)
                else:
                    # Если точек нет, создаем пустой эмбеддинг
                    batch_sparse_embeddings.append(torch.zeros(
                        (1, 0, self.sam.prompt_encoder.embed_dim), 
                        device=images.device
                    ))
            
            # Собираем все sparse embeddings в один тензор
            sparse_embeddings = torch.cat(batch_sparse_embeddings, dim=0)
        
        # Декодируем маски
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False  # Возвращаем только одну маску
        )
        
        # Преобразуем маски к нужному размеру
        masks = []
        for i in range(B):
            if original_sizes is not None:
                # Используем оригинальный размер для upscale
                orig_h, orig_w = original_sizes[i]
                mask = torch.nn.functional.interpolate(
                    low_res_masks[i].unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                mask = low_res_masks[i]
            masks.append(mask)
        
        # Собираем все маски в один тензор [B, 1, H, W]
        masks = torch.stack(masks, dim=0)
        
        return masks

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
    patience = 10  # Количество эпох для early stopping
    patience_counter = 0
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
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

            for batch in dataloaders[phase]:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                prompt_points = batch['prompt_points']
                prompt_labels = batch['prompt_labels']
                original_sizes = batch['original_size']
                
                # Переносим prompt точки и метки на устройство, если они есть
                batch_prompt_points = []
                batch_prompt_labels = []
                for pts, lbls in zip(prompt_points, prompt_labels):
                    batch_prompt_points.append(pts.to(device))
                    batch_prompt_labels.append(lbls.to(device))

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Прямой проход модели
                    outputs = model(
                        images, 
                        prompt_points=batch_prompt_points,
                        prompt_labels=batch_prompt_labels,
                        original_sizes=original_sizes
                    )
                    
                    # Вычисляем потери
                    loss = criterion(outputs, masks)

                    # Считаем IoU
                    batch_iou = iou_score(outputs, masks)
                    running_iou += batch_iou.item() * images.size(0)
                    running_loss += loss.item() * images.size(0)
                    num_samples += images.size(0)

                    if phase == 'train':
                        loss.backward()
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

            epoch_loss = running_loss / num_samples
            epoch_iou = running_iou / num_samples

            print(f"{phase} Loss: {epoch_loss:.4f} | {phase} IoU: {epoch_iou:.4f}")

            # Сохраняем лучшую модель и обновляем scheduler
            if phase == 'val':
                scheduler.step(epoch_iou)
                if epoch_iou > best_iou:
                    best_iou = epoch_iou
                    torch.save(model.state_dict(), "best_sam2_model.pth")
                    print("Модель сохранена (лучшая IoU).")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping после {epoch + 1} эпох")
                        return model

    print("\nОбучение завершено.")
    print(f"Лучшая валидационная IoU: {best_iou:.4f}")
    return model

# --------------------------------------------------
# 7. ИНФЕРЕНС (ПРОГНОЗ)
# --------------------------------------------------
def predict_image(model, image_path, device, threshold=0.5):
    """
    Функция для прогноза маски на одном изображении.
    """
    model.eval()
    
    # Загрузка и предобработка изображения
    image = np.array(Image.open(image_path).convert("RGB"))
    resize_transform = ResizeLongestSide(IMAGE_SIZE)
    
    # Применяем resize трансформацию SAM2
    image_resized = resize_transform.apply_image(image)
    original_size = torch.tensor([image.shape[0], image.shape[1]], dtype=torch.int)
    
    # Преобразуем в тензор
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Автоматически генерируем точки подсказки, если это возможно
    # (в реальном сценарии здесь можно использовать автоматический детектор объектов
    # или позволить пользователю указать точки подсказки)
    
    # Для простоты примера возьмем несколько точек в центре изображения
    h, w = image_resized.shape[:2]
    prompt_points = torch.tensor([[[w/2, h/2]]], dtype=torch.float).to(device)
    prompt_labels = torch.tensor([[1]], dtype=torch.int).to(device)
    
    with torch.no_grad():
        # Получаем предсказание от модели
        outputs = model(
            image_tensor, 
            prompt_points=[prompt_points[0]], 
            prompt_labels=[prompt_labels[0]],
            original_sizes=[original_size]
        )
        
        # Применяем сигмоиду и порог
        pred_mask = (torch.sigmoid(outputs) > threshold).float()
    
    # Приводим маску к numpy
    pred_mask = pred_mask.squeeze().cpu().numpy()
    
    return pred_mask

def export_model_onnx(model, device):
    """
    Экспортирует модель SAM2 в формат ONNX для оптимизации инференса.
    """
    model.eval()
    
    # Создаем пример входных данных
    dummy_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_points = torch.tensor([[[0.5, 0.5]]], dtype=torch.float).to(device)
    dummy_labels = torch.tensor([[1]], dtype=torch.int).to(device)
    dummy_size = torch.tensor([[IMAGE_SIZE, IMAGE_SIZE]], dtype=torch.int)
    
    # Определяем путь для сохранения ONNX модели
    onnx_path = "sam2_model.onnx"
    
    # Определяем функцию для экспорта
    def export_model(model, image, points, labels, sizes):
        return model(image, [points[0]], [labels[0]], [sizes[0]])
    
    # Экспортируем модель в ONNX
    torch.onnx.export(
        model,
        (dummy_image, dummy_points, dummy_labels, dummy_size),
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['image', 'points', 'labels', 'original_size'],
        output_names=['mask'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'points': {0: 'batch_size', 1: 'num_points'},
            'labels': {0: 'batch_size', 1: 'num_points'},
            'mask': {0: 'batch_size'}
        }
    )
    
    print(f"Модель экспортирована и сохранена как '{onnx_path}'")
    return onnx_path

# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    # Создаем и загружаем датасеты
    train_dataset = BubbleDatasetSAM(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
    val_dataset = BubbleDatasetSAM(VAL_IMAGES_DIR, VAL_MASKS_DIR)

    # Создаем DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }

    # Проверяем наличие предобученной модели SAM2
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"Ошибка: Веса модели SAM2 не найдены по пути {SAM_CHECKPOINT}")
        print("Пожалуйста, загрузите веса модели SAM2 с официального репозитория.")
        return

    # Инициализируем модель SAM2 с предобученными весами
    model = SAM2Wrapper(SAM_CHECKPOINT, SAM_TYPE)
    model = model.to(device)

    # Выводим информацию о параметрах модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,} ({trainable_params/total_params:.2%})")

    # Определяем функцию потерь и оптимизатор
    criterion = BCEDiceLoss()
    # Используем AdamW только для обучаемых параметров
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # Обучаем модель
    model = train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS, device)
    
    # Экспортируем модель в ONNX формат
    onnx_path = export_model_onnx(model, device)

    # Пример инференса на одном изображении
    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        print("Запускаем инференс на тестовом изображении...")
        pred_mask = predict_image(model, test_image_path, device)
        print("Предсказанная маска имеет форму:", pred_mask.shape)
        
        # Сохраняем предсказанную маску для визуализации
        mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        mask_img.save('predicted_mask.png')
        print("Маска сохранена как 'predicted_mask.png'")
    else:
        print("test_image.jpg не найден, пропускаем инференс.")

if __name__ == '__main__':
    main() 