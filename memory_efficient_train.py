import os
import torch
from ultralytics import YOLO

# Параметры обучения с оптимизацией памяти
BATCH_SIZE = 1  # Минимальный размер батча
NUM_EPOCHS = 5  # Меньше эпох для тестирования
IMAGE_SIZE = 480  # Уменьшенный размер изображения (оригинал 640)
YOLO_MODEL = 'yolo11n-seg.pt'  # Локальная модель
YOLO_DIR = 'yolo_data'

# Устанавливаем параметры для экономии памяти
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Для лучшей отладки CUDA
torch.backends.cudnn.benchmark = False  # Экономит память

def train_yolo_model():
    """
    Функция для обучения модели YOLO с оптимизацией памяти
    """
    try:
        print(f"Используем устройство: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Устанавливаем ограничения по памяти для CUDA если доступно
        if torch.cuda.is_available():
            # Освобождаем кэш CUDA
            torch.cuda.empty_cache()
            
            # Проверяем доступную память GPU
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Доступная память GPU: {gpu_mem:.2f} ГБ")
        
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
        
        # Загружаем модель с осторожностью по памяти
        print(f"Загружаем модель {YOLO_MODEL}...")
        model = YOLO(YOLO_MODEL)
        print("Модель загружена успешно!")
        
        # Запускаем обучение с оптимизацией памяти
        print("Начинаем обучение...")
        results = model.train(
            data=yaml_path,
            epochs=NUM_EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            workers=1,  # Снижаем количество рабочих процессов
            patience=3,  # Раннее останавливание
            device='0' if torch.cuda.is_available() else 'cpu',
            project='yolo_runs',
            name='bubbles_segmentation_memory_optimized',
            pretrained=True,
            optimizer='AdamW',
            lr0=1e-4,  # Снижаем скорость обучения
            lrf=0.01,
            weight_decay=1e-4,
            save=True,
            exist_ok=True,
            verbose=True,
            cache=False,  # Отключаем кэширование данных
            half=True,  # Включаем режим половинной точности для экономии памяти
            fraction=0.5,  # Используем только половину данных для обучения
            overlap_mask=False,  # Отключаем перекрытие масок
            mask_ratio=16,  # Увеличиваем соотношение размеров маски (уменьшает размер маски)
            close_mosaic=0  # Отключаем мозаику сразу
        )
        
        return results
    
    except Exception as e:
        print(f"Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    train_yolo_model() 