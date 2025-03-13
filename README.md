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

## Оптимизированная модель

Проект поддерживает оптимизированные версии моделей с использованием TorchScript и ONNX:

### TorchScript модель

```python
# Загрузка оптимизированной модели
model = torch.jit.load("optimized_model.pt")
model.to(device)

# Использование модели для предсказаний
with torch.no_grad():
    outputs = model(images)
    predictions = torch.sigmoid(outputs) > 0.5
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