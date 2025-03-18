# FrothModel

Проект по сегментации пузырьков на изображениях с использованием нейронных сетей и PyTorch.

## Описание

FrothModel - это инструмент для автоматической сегментации пузырьков (пены) на изображениях с использованием архитектуры U-Net и предварительно обученных энкодеров. Проект использует библиотеку segmentation_models_pytorch для эффективной реализации сегментационных моделей.

## Структура проекта

- `training.py` - скрипт для обучения модели сегментации
- `testing.py` - скрипт для оценки производительности обученной модели
- `convert_to_onnx.py` - скрипт для конвертации PyTorch модели в формат ONNX
- `best_model.pth` - файл с сохраненными весами лучшей модели
- `optimized_model.pt` - оптимизированная версия модели (TorchScript)
- `model.onnx` - модель в формате ONNX
- `model_optimized.onnx` - оптимизированная модель в формате ONNX
- `requirements.txt` - список зависимостей
- `DataSet/` - каталог с данными для обучения и тестирования
  - `train/train_images/` - изображения для обучения
  - `train/train_masks/` - маски для обучения
  - `valid/valid_images/` - изображения для валидации
  - `valid/valid_masks/` - маски для валидации
  - `test/test_images/` - изображения для тестирования
  - `test/test_masks/` - маски для тестирования


## Датасет
https://www.kaggle.com/datasets/obobojk/froth-bubbles

## Установка и запуск

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Nestert/FrothModel.git
cd FrothModel
```

2. Зависимости:
```bash
pip install -r requirements.txt
```

3. Обучение модели:
```bash
python training.py
```

4. Тестирование модели:
```bash
python testing.py
```

5. Конвертация модели в ONNX формат:
```bash
python convert_to_onnx.py --model best_model.pth --output model.onnx
```

## Технические детали

- Архитектура: U-Net с MobileNetV3 Large в качестве энкодера
- Функция потерь: Binary Cross-Entropy with Logits Loss
- Оптимизатор: Adam
- Размер изображений: 512x512
- Метрика оценки: IoU (Intersection over Union)

## Результаты

### Первоначальные результаты
Epoch 47/50
----------
train Loss: 0.1245 | train IoU: 0.7431
val Loss: 0.1255 | val IoU: 0.7497

Test IoU: 0.6947

### 14.03.25
Epoch 40/50
----------
train Loss: 0.1436 | train IoU: 0.7077
val Loss: 0.1342 | val IoU: 0.7351

Итоговый Test best_model IoU: 0.6851
Итоговый Test optimized_model IoU: 0.6823

### 17.03.25 - Результаты после оптимизации
Исследование различных методов оптимизации модели, включая pruning, квантизацию и дистилляцию знаний:

--------------------------------------------------
Модель               | IoU
--------------------------------------------------
teacher              | 0.6807 (оригинальная модель)
student              | 0.6631 (дистилляция знаний)
optimized            | 0.6631 (TorchScript оптимизация)
--------------------------------------------------

## Методы оптимизации

Для повышения эффективности модели и уменьшения её размера были использованы следующие методы оптимизации:

### 1. Pruning (обрезка параметров)
Pruning использует технику L1 unstructured pruning для удаления наименее важных весов в сети, что позволяет уменьшить размер модели при минимальной потере точности.

### 2. Квантизация
Динамическая квантизация преобразует веса из формата float32 в int8, что значительно сокращает размер модели и ускоряет инференс, особенно на CPU.

### 3. Дистилляция знаний
Обучение меньшей "student" модели на основе более крупной "teacher" модели. Student модель имитирует поведение teacher модели, что позволяет достичь сравнимой точности при меньшем числе параметров и более высокой скорости инференса.

### 4. TorchScript оптимизация
Конвертация модели в TorchScript формат позволяет оптимизировать вычислительный граф и обеспечивает более эффективное выполнение модели в производственной среде.

### 5. ONNX конвертация
Экспорт модели в ONNX формат позволяет использовать модель в различных фреймворках и на различных устройствах, обеспечивая кроссплатформенную совместимость.

## Оптимизированная модель

Проект поддерживает различные типы оптимизированных моделей:

### Применение пост-обучающей оптимизации

Для применения pruning, квантизации, дистилляции знаний и других методов оптимизации используйте скрипт `post_training_optimization.py`:

```bash
# Базовое применение всех оптимизаций
python post_training_optimization.py
```

Этот скрипт позволяет:
- Загрузить предварительно обученную teacher модель
- Применить pruning к модели
- Выполнить динамическую квантизацию
- Обучить student модель с использованием дистилляции знаний
- Оптимизировать модель с помощью TorchScript
- Конвертировать модель в формат ONNX

### Использование TorchScript модели

```python
# Загрузка оптимизированной модели
model = torch.jit.load("models/optimized_model.pt")
model.to(device)

# Использование модели для предсказаний
with torch.no_grad():
    outputs = model(images)
    predictions = torch.sigmoid(outputs) > 0.5
```

### Использование обрезанной (pruned) модели

```python
# Загрузка pruned модели
model = smp.Unet(
    encoder_name="timm-mobilenetv3_large_100",
    encoder_weights=None,  # не загружаем предварительно обученные веса
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load("models/teacher_model_pruned.pth"))
model.to(device)
```

### Использование квантованной модели

```python
# Загрузка квантованной модели
model = smp.Unet(
    encoder_name="timm-mobilenetv3_large_100",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load("models/teacher_model_quantized.pth"))
model.to(device)
```

## Конвертация в ONNX

Для конвертации модели из формата PyTorch в ONNX используйте скрипт `convert_to_onnx.py`:

```bash
# Базовое использование
python convert_to_onnx.py --model best_model.pth --output model.onnx

# С указанием пути для оптимизированной версии
python convert_to_onnx.py --model best_model.pth --output model.onnx --optimized_output model_optimized.onnx

# С тестированием на конкретном изображении
python convert_to_onnx.py --model best_model.pth --test_image test.jpg

# С тестированием производительности
python convert_to_onnx.py --model best_model.pth --benchmark

# С указанием размера батча (по умолчанию 1)
python convert_to_onnx.py --model best_model.pth --batch_size 4
```

## Требования

- Python 3.6+
- PyTorch 1.7.0+
- torchvision 0.8.0+
- Pillow 8.0.0+
- NumPy 1.19.0+
- segmentation-models-pytorch 0.2.0+
- albumentations 1.0.0+
- onnx 1.12.0+
- onnxruntime 1.12.0+ 