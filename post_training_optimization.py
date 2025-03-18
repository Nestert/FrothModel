import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --------------------------------------------------
# 1. ПАРАМЕТРЫ
# --------------------------------------------------
DATA_DIR = 'DataSet'  # Корневая папка с данными
MODELS_DIR = 'models'  # Папка для сохранения моделей
# Создаем директорию для моделей, если она не существует
os.makedirs(MODELS_DIR, exist_ok=True)

IMAGE_SIZE = (512, 512)  # Размер изображений

# Определяем размеры для teacher и student модели
TEACHER_BACKBONE = "timm-mobilenetv3_large_100"
STUDENT_BACKBONE = "timm-mobilenetv3_small_100"
ENCODER_WEIGHTS = "imagenet"

# --------------------------------------------------
# 2. ТРАНСФОРМАЦИИ ДЛЯ ИНФЕРЕНСА
# --------------------------------------------------
def get_transforms(phase='val'):
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
    elif phase == 'val':
        return A.Compose([
            A.LongestMaxSize(max_size=max(IMAGE_SIZE)),
            A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1], border_mode=0),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return None

# --------------------------------------------------
# 3. ФУНКЦИИ ОПТИМИЗАЦИИ
# --------------------------------------------------
def apply_pruning(model, amount=0.3):
    """
    Применяет L1 unstructured pruning к свёрточным слоям.
    """
    import torch_pruning as tp
    
    model.eval()
    
    # Создаем экземпляр pruner с использованием MagnitudePruner (pruning по величине весов)
    example_inputs = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создаем pruner
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=tp.importance.MagnitudeImportance(p=1),  # L1 norm
        pruning_ratio=amount,
    )
    
    # Выполняем pruning
    pruner.step()
    
    print(f"Model pruned successfully with ratio: {amount}")
    return model

def apply_dynamic_quantization(model):
    """
    Динамическое квантование (работает в основном для Linear слоёв).
    Здесь используется для демонстрации, хотя эффект на свёрточных слоях может быть незначителен.
    """
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return quantized_model

def optimize_model(model, device):
    """
    Оптимизирует модель с помощью TorchScript для более быстрого инференса.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model_path = os.path.join(MODELS_DIR, "optimized_model.pt")
    torch.jit.save(traced_model, traced_model_path)
    print(f"Model optimized and saved as '{traced_model_path}'")
    return traced_model

def convert_to_onnx(model, device):
    """
    Конвертирует модель в формат ONNX для использования в других фреймворках.
    """
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

def predict_image(model, image_path, device, transform=None, threshold=0.5):
    """
    Делает предсказание на одном изображении.
    """
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

# --------------------------------------------------
# 4. ОБУЧЕНИЕ STUDENT МОДЕЛИ С ДИСТИЛЛЯЦИЕЙ
# --------------------------------------------------
def train_student_with_distillation(teacher_model, device, num_epochs=50, lr=1e-3, batch_size=8, alpha=0.5, temperature=2.0):
    """
    Обучает student модель с помощью дистилляции от teacher модели.
    Эта функция предполагает, что датасеты уже созданы и находятся в соответствующих папках.
    """
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    from training import BubbleDataset, BCEDiceLoss, train_model_student

    # Создаём датасеты и DataLoader'ы
    train_dataset = BubbleDataset(
        os.path.join(DATA_DIR, 'train/train_images'), 
        os.path.join(DATA_DIR, 'train/train_masks'), 
        transform=get_transforms('train')
    )
    val_dataset = BubbleDataset(
        os.path.join(DATA_DIR, 'valid/valid_images'), 
        os.path.join(DATA_DIR, 'valid/valid_masks'), 
        transform=get_transforms('val')
    )
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    student_model = smp.Unet(
        encoder_name=STUDENT_BACKBONE,
        encoder_weights=ENCODER_WEIGHTS,  
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    criterion = BCEDiceLoss()
    optimizer_student = optim.AdamW(student_model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Замораживаем teacher модель для дистилляции
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    student_model = train_model_student(
        student_model, teacher_model, dataloaders, criterion, 
        optimizer_student, num_epochs, device, alpha=alpha, temperature=temperature
    )
    
    torch.save(student_model.state_dict(), os.path.join(MODELS_DIR, "student_model.pth"))
    print("Student модель сохранена.")
    
    return student_model

# --------------------------------------------------
# 5. ОСНОВНАЯ ФУНКЦИЯ
# --------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Загрузка обученной teacher модели
    teacher_model_path = os.path.join(MODELS_DIR, "teacher_model.pth")
    if not os.path.exists(teacher_model_path):
        print(f"Ошибка: Файл модели {teacher_model_path} не найден.")
        return
    
    teacher_model = smp.Unet(
        encoder_name=TEACHER_BACKBONE,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    teacher_model.load_state_dict(torch.load(teacher_model_path))
    print("Teacher модель загружена.")

    # 2. Применяем pruning к teacher модели
    print("\nПрименение pruning к teacher модели...")
    teacher_model = apply_pruning(teacher_model, amount=0.3)
    torch.save(teacher_model.state_dict(), os.path.join(MODELS_DIR, "teacher_model_pruned.pth"))
    print("Teacher модель после pruning сохранена.")

    # 3. Применяем динамическое квантование
    print("\nПрименение динамической квантования к teacher модели...")
    teacher_model_quantized = apply_dynamic_quantization(teacher_model)
    torch.save(teacher_model_quantized.state_dict(), os.path.join(MODELS_DIR, "teacher_model_quantized.pth"))
    print("Teacher модель после квантования сохранена.")

    # 4. (Опционально) Обучение student модели с дистилляцией
    train_student = input("Хотите ли вы обучить student модель с дистилляцией? (y/n): ").lower() == 'y'
    if train_student:
        print("\nОбучение student модели с дистилляцией...")
        student_model = train_student_with_distillation(teacher_model, device)
    else:
        # Если не обучаем student модель, то работаем с pruned teacher моделью
        student_model = teacher_model

    # 5. Оптимизация модели для инференса (TorchScript)
    optimized_model = optimize_model(student_model, device)

    # 6. Конвертация модели в формат ONNX
    convert_to_onnx(student_model, device)

    # 7. Пример инференса на одном изображении
    test_image_path = input("Введите путь к тестовому изображению (или нажмите Enter для пропуска): ")
    if test_image_path and os.path.exists(test_image_path):
        pred_mask = predict_image(student_model, test_image_path, device, transform=get_transforms('val'))
        print("Предсказанная маска имеет форму:", pred_mask.shape)
        
        # Сохранение предсказанной маски
        mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8))
        mask_save_path = os.path.splitext(test_image_path)[0] + "_pred_mask.png"
        mask_image.save(mask_save_path)
        print(f"Предсказанная маска сохранена как {mask_save_path}")
    else:
        print("Тестовое изображение не указано или не найдено, пропускаем инференс.")

if __name__ == '__main__':
    main() 