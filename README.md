# FrothModel

Проект по сегментации пузырьков на изображениях с использованием нейронных сетей и PyTorch.

## Описание

FrothModel - это инструмент для автоматической сегментации пузырьков (пены) на изображениях с использованием архитектуры U-Net и предварительно обученных энкодеров. Проект использует библиотеку segmentation_models_pytorch для эффективной реализации сегментационных моделей.

## Структура проекта

- `training.py` - скрипт для обучения модели сегментации
- `testing.py` - скрипт для оценки производительности обученной модели
- `best_model.pth` - файл с сохраненными весами лучшей модели
- `requirements.txt` - список зависимостей
- `DataSet/` - каталог с данными для обучения и тестирования
  - `train/train_images/` - изображения для обучения
  - `train/train_masks/` - маски для обучения
  - `valid/valid_images/` - изображения для валидации
  - `valid/valid_masks/` - маски для валидации


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

## Технические детали

- Архитектура: U-Net с ResNet34 в качестве энкодера
- Функция потерь: Binary Cross-Entropy with Logits Loss
- Оптимизатор: Adam
- Размер изображений: 512x512
- Метрика оценки: IoU (Intersection over Union)

## Result:
Epoch 47/50
----------
train Loss: 0.1245 | train IoU: 0.7431
val Loss: 0.1255 | val IoU: 0.7497

Test IoU: 0.6947

## Требования

- Python 3.6+
- PyTorch 1.7.0+
- torchvision 0.8.0+
- Pillow 8.0.0+
- NumPy 1.19.0+
- segmentation-models-pytorch 0.2.0+
- albumentations 1.0.0+ 