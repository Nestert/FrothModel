import os
import torch
from ultralytics import YOLO

# Параметры обучения
BATCH_SIZE = 4
NUM_EPOCHS = 5  # Еще меньше эпох для быстрого тестирования
IMAGE_SIZE = 640
YOLO_MODEL = 'yolo11n-seg.pt'  # Локальная модель
YOLO_DIR = 'yolo_data'

# Отключаем ненужные проверки сети
os.environ['ULTRALYTICS_OFFLINE'] = '1'

def train_yolo_model():
    """
    Функция для обучения модели YOLO
    """
    try:
        print(f"Используем устройство: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Проверяем наличие модели
        if not os.path.exists(YOLO_MODEL):
            print(f"Модель {YOLO_MODEL} не найдена!")
            return None
        
        print(f"Найдена модель {YOLO_MODEL}, размер: {os.path.getsize(YOLO_MODEL) / (1024*1024):.2f} МБ")
        
        # Проверяем наличие данных
        yaml_path = os.path.join(YOLO_DIR, 'data.yaml')
        if not os.path.exists(yaml_path):
            print(f"Файл конфигурации {yaml_path} не найден!")
            return None
            
        print(f"Найден файл конфигурации {yaml_path}")
        
        # Загружаем модель
        print(f"Загружаем модель {YOLO_MODEL}...")
        model = YOLO(YOLO_MODEL)
        print("Модель загружена успешно!")
        
        # Запускаем обучение
        print("Начинаем обучение...")
        results = model.train(
            data=yaml_path,
            epochs=NUM_EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            patience=3,  # Раннее останавливание
            device='0' if torch.cuda.is_available() else 'cpu',
            project='yolo_runs',
            name='bubbles_segmentation',
            pretrained=True,
            optimizer='AdamW',
            lr0=1e-3,
            lrf=0.01,
            weight_decay=1e-4,
            save=True,
            exist_ok=True,
            verbose=True
        )
        
        return results
    
    except Exception as e:
        print(f"Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    train_yolo_model() 