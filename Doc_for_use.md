# Использование оптимизированных моделей сегментации пены в других проектах

В данном документе описано, как использовать модели, созданные и оптимизированные с помощью скрипта `post_training_optimization.py`, в ваших собственных проектах.

## Доступные модели

После выполнения скрипта оптимизации у вас будут следующие модели:

1. **Оригинальная Teacher-модель** (`teacher_model.pth`) - полноразмерная модель Unet с backbone MobileNetV3 Large.
2. **Прунированная Teacher-модель** (`teacher_model_pruned.pth`) - модель с удалёнными избыточными весами.
3. **Квантованная Teacher-модель** (`teacher_model_quantized.pth`) - модель с пониженной точностью весов.
4. **Student-модель** (`student_model.pth`) - если вы выбрали её обучение, меньшая модель на основе MobileNetV3 Small.
5. **Оптимизированная TorchScript-модель** (`optimized_model.pt`) - модель, оптимизированная для быстрого инференса.
6. **ONNX-модель** (`model.onnx`) - модель в кросс-платформенном формате ONNX.

## Требования

Для использования моделей вам потребуются следующие библиотеки:

```python
import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
```

## Загрузка и использование моделей

### 1. Загрузка PyTorch-моделей (.pth)

```python
import torch
import segmentation_models_pytorch as smp

# Параметры модели
BACKBONE = "timm-mobilenetv3_large_100"  # или "timm-mobilenetv3_small_100" для student-модели
ENCODER_WEIGHTS = "imagenet"
IMAGE_SIZE = (512, 512)

# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name=BACKBONE,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=1,
    activation=None
).to(device)

# Загрузка весов
model.load_state_dict(torch.load("models/teacher_model_pruned.pth", map_location=device))
model.eval()
```

### 2. Загрузка квантованной модели

```python
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn

# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name=BACKBONE,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=1,
    activation=None
).to(device)

# Применяем квантование
quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# Загрузка весов
quantized_model.load_state_dict(torch.load("models/teacher_model_quantized.pth", map_location=device))
quantized_model.eval()
```

### 3. Загрузка TorchScript-модели (.pt)

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("models/optimized_model.pt", map_location=device)
model.eval()
```

### 4. Загрузка ONNX-модели

```python
import onnxruntime as ort

# Создание ONNX Runtime сессии
ort_session = ort.InferenceSession("models/model.onnx")
```

## Препроцессинг изображений

Для предобработки изображений перед инференсом используйте следующие трансформации:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = (512, 512)

transforms = A.Compose([
    A.LongestMaxSize(max_size=max(IMAGE_SIZE)),
    A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1], border_mode=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

## Инференс

### Инференс с PyTorch-моделями

```python
import numpy as np
from PIL import Image
import torch

def predict_image(model, image_path, device, transforms, threshold=0.5):
    # Загрузка и препроцессинг изображения
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = transforms(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Инференс
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        pred_mask = (output > threshold).float()
    
    return pred_mask.squeeze().cpu().numpy()

# Пример использования
mask = predict_image(model, "path/to/image.jpg", device, transforms)

# Сохранение маски
mask_image = Image.fromarray((mask * 255).astype(np.uint8))
mask_image.save("predicted_mask.png")
```

### Инференс с ONNX-моделью

```python
import numpy as np
from PIL import Image
import onnxruntime as ort

def predict_image_onnx(ort_session, image_path, transforms, threshold=0.5):
    # Загрузка и препроцессинг изображения
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = transforms(image=image)
    input_tensor = augmented['image'].unsqueeze(0).numpy()  # ONNX принимает numpy массивы
    
    # Инференс
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    output = ort_outputs[0]
    
    # Постпроцессинг
    output = 1 / (1 + np.exp(-output))  # sigmoid
    pred_mask = (output > threshold).astype(np.float32)
    
    return pred_mask.squeeze()

# Пример использования
mask = predict_image_onnx(ort_session, "path/to/image.jpg", transforms)

# Сохранение маски
mask_image = Image.fromarray((mask * 255).astype(np.uint8))
mask_image.save("predicted_mask.png")
```

## Примечания по производительности

1. **PyTorch-модели (.pth)**: 
   - Наивысшая точность для teacher-модели
   - Прунированная модель имеет меньший размер и более быструю работу
   - Подходит для использования на GPU с PyTorch

2. **Квантованная модель**:
   - Меньший размер файла
   - Может быть быстрее на CPU
   - Небольшая потеря в точности

3. **TorchScript-модель (.pt)**:
   - Оптимизирована для инференса
   - Работает быстрее обычной PyTorch-модели
   - Требует PyTorch для запуска

4. **ONNX-модель**:
   - Кросс-платформенный формат
   - Можно использовать без PyTorch через ONNX Runtime
   - Подходит для развертывания на разных устройствах и платформах
   - Оптимальна для продакшн-систем

5. **Student-модель**:
   - Значительно меньше и быстрее teacher-модели
   - Небольшая потеря в точности компенсируется скоростью
   - Рекомендуется для устройств с ограниченными ресурсами

## Пример полного пайплайна

```python
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
model = torch.jit.load("models/optimized_model.pt", map_location=device)
model.eval()

# Настройка трансформаций
IMAGE_SIZE = (512, 512)
transforms = A.Compose([
    A.LongestMaxSize(max_size=max(IMAGE_SIZE)),
    A.PadIfNeeded(min_height=IMAGE_SIZE[0], min_width=IMAGE_SIZE[1], border_mode=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Функция инференса
def segment_foam(image_path, threshold=0.5):
    # Загрузка и препроцессинг изображения
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = transforms(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Инференс
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        pred_mask = (output > threshold).float()
    
    # Конвертация в изображение
    mask_array = pred_mask.squeeze().cpu().numpy()
    mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))
    
    return mask_image

# Пример использования
foam_mask = segment_foam("path/to/foam_image.jpg")
foam_mask.save("foam_mask.png")
``` 