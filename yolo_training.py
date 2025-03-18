import os
import yaml
import numpy as np
from glob import glob
from PIL import Image
import shutil

import torch
from ultralytics import YOLO

# --------------------------------------------------
# 1. ПАРАМЕТРЫ
# --------------------------------------------------
DATA_DIR = 'DataSet'  # Корневая папка с данными
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train/train_images')
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, 'train/train_masks')
VAL_IMAGES_DIR = os.path.join(DATA_DIR, 'valid/valid_images')
VAL_MASKS_DIR = os.path.join(DATA_DIR, 'valid/valid_masks')

# Параметры обучения
BATCH_SIZE = 16  # YOLO хорошо работает с большими батчами
NUM_EPOCHS = 50
IMAGE_SIZE = 640  # YOLO обычно использует размер 640
YOLO_MODEL = 'yolov11n-seg.pt'  # Опции: yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt

# Директория для YOLO данных
YOLO_DIR = 'yolo_data'
YOLO_IMAGES_DIR = os.path.join(YOLO_DIR, 'images')
YOLO_LABELS_DIR = os.path.join(YOLO_DIR, 'labels')

# --------------------------------------------------
# 2. ПОДГОТОВКА ДАННЫХ ДЛЯ YOLO
# --------------------------------------------------
def create_yolo_dataset_structure():
    """
    Создает структуру директорий для YOLO: train/val/images/labels
    """
    # Создаем основные директории
    os.makedirs(os.path.join(YOLO_DIR, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, 'labels/val'), exist_ok=True)

def mask_to_yolo_polygon(mask, min_area=10):
    """
    Преобразование бинарной маски в полигоны YOLO.
    Результат: массив полигонов в формате YOLO [[x1, y1, x2, y2, ..., xn, yn], ...]
    """
    import cv2
    # Преобразуем маску в uint8 для cv2
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Находим контуры
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    height, width = mask.shape
    
    for contour in contours:
        # Фильтруем очень маленькие контуры
        if cv2.contourArea(contour) < min_area:
            continue
        
        # Нормализуем координаты контура от 0 до 1
        polygon = []
        for point in contour.reshape(-1, 2):
            x, y = point
            polygon.extend([x / width, y / height])
        
        # Упрощаем полигон, если он слишком сложный
        if len(polygon) > 200:  # Если полигон содержит более 100 точек
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = []
            for point in approx.reshape(-1, 2):
                x, y = point
                polygon.extend([x / width, y / height])
        
        polygons.append(polygon)
    
    return polygons

def convert_dataset_to_yolo_format():
    """
    Конвертирует датасет из исходного формата в формат YOLO.
    """
    create_yolo_dataset_structure()
    
    # Обрабатываем тренировочные данные
    process_dataset_split('train', TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
    
    # Обрабатываем валидационные данные
    process_dataset_split('val', VAL_IMAGES_DIR, VAL_MASKS_DIR)
    
    # Создаем файл data.yaml для YOLO
    create_data_yaml()

def process_dataset_split(split, images_dir, masks_dir):
    """
    Обрабатывает один сплит (train или val) датасета.
    """
    image_paths = sorted(glob(os.path.join(images_dir, '*')))
    mask_paths = sorted(glob(os.path.join(masks_dir, '*')))
    
    assert len(image_paths) == len(mask_paths), f"Число изображений и масок не совпадает в {split}!"
    
    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Имя файла без расширения
        img_filename = os.path.basename(img_path)
        mask_filename = os.path.basename(mask_path)
        base_name = os.path.splitext(img_filename)[0]
        
        # Копируем изображение в директорию YOLO
        dest_img_path = os.path.join(YOLO_DIR, f'images/{split}/{img_filename}')
        shutil.copy(img_path, dest_img_path)
        
        # Загружаем маску
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)
        
        # Конвертируем маску в полигоны YOLO
        polygons = mask_to_yolo_polygon(mask)
        
        # Записываем полигоны в файл .txt (формат YOLO)
        label_path = os.path.join(YOLO_DIR, f'labels/{split}/{base_name}.txt')
        with open(label_path, 'w') as f:
            for polygon in polygons:
                polygon_str = ' '.join([f"{p:.6f}" for p in polygon])
                # 0 - это класс (у нас только один класс - пузырек)
                f.write(f"0 {polygon_str}\n")
        
        if i % 100 == 0:
            print(f"Обработано {i}/{len(image_paths)} изображений в {split}")

def create_data_yaml():
    """
    Создает файл data.yaml для настройки YOLO
    """
    data = {
        'path': os.path.abspath(YOLO_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'bubble'},  # Пузырек - единственный класс
        'nc': 1  # Количество классов
    }
    
    with open(os.path.join(YOLO_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

# --------------------------------------------------
# 3. ОБУЧЕНИЕ МОДЕЛИ
# --------------------------------------------------
def train_yolo_model():
    """
    Обучает YOLOv8-seg на подготовленном датасете.
    """
    # Загружаем предобученную модель
    model = YOLO(YOLO_MODEL)
    
    # Запускаем обучение
    results = model.train(
        data=os.path.join(YOLO_DIR, 'data.yaml'),
        epochs=NUM_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=10,  # Early stopping
        device='0' if torch.cuda.is_available() else 'cpu',
        project='yolo_runs',
        name='bubbles_segmentation',
        pretrained=True,
        optimizer='AdamW',  # Тот же оптимизатор, что и в исходном коде
        lr0=1e-3,
        lrf=0.01,
        weight_decay=1e-4,  # L2 регуляризация как в исходном коде
        save=True,
        save_period=5,  # Сохранять каждые 5 эпох
        exist_ok=True
    )
    
    return results

# --------------------------------------------------
# 4. ИНФЕРЕНС (ПРОГНОЗ)
# --------------------------------------------------
def predict_image(model, image_path, threshold=0.5):
    """
    Функция для прогноза маски на одном изображении.
    """
    # Запускаем прогноз
    results = model.predict(
        source=image_path,
        conf=threshold,
        save=True,
        project='yolo_predictions',
        name='bubbles'
    )
    
    return results

# --------------------------------------------------
# 5. ЭКСПОРТ МОДЕЛИ
# --------------------------------------------------
def export_model(model, format='onnx'):
    """
    Экспортирует обученную модель в различные форматы.
    Возможные форматы: onnx, openvino, engine (для TensorRT), coreml
    """
    exported_model = model.export(format=format)
    return exported_model

# --------------------------------------------------
# 6. MAIN
# --------------------------------------------------
def main():
    # Проверяем наличие GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")
    
    # Конвертируем датасет в формат YOLO
    print("Конвертируем датасет в формат YOLO...")
    convert_dataset_to_yolo_format()
    
    # Обучаем модель
    print("Запускаем обучение...")
    results = train_yolo_model()
    
    # Загружаем лучшую модель
    best_model_path = os.path.join('yolo_runs/bubbles_segmentation/weights', 'best.pt')
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        
        # Экспортируем модель в ONNX
        print("Экспортируем модель в ONNX...")
        exported_model = export_model(best_model, format='onnx')
        
        # Пример инференса на тестовом изображении
        test_image_path = "test_image.jpg"
        if os.path.exists(test_image_path):
            print("Запускаем инференс на тестовом изображении...")
            results = predict_image(best_model, test_image_path)
            print(f"Результаты инференса сохранены в директории: {results[0].save_dir}")
        else:
            print("test_image.jpg не найден, пропускаем инференс.")
    else:
        print("Не удалось найти лучшую модель.")

if __name__ == '__main__':
    main() 