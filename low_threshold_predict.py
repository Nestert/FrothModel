import os
import torch
from ultralytics import YOLO
import glob

# Параметры
MODEL_PATH = 'yolo11n-seg.pt'
CONF_THRESHOLD = 0.05  # Очень низкий порог уверенности для обнаружения пузырьков
DATA_DIR = 'DataSet'
OUTPUT_DIR = 'predictions_low_threshold'

def predict_on_images():
    """
    Запускает инференс на тестовых изображениях с низким порогом обнаружения
    """
    try:
        print(f"Используем устройство: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Проверяем наличие модели
        if not os.path.exists(MODEL_PATH):
            print(f"Модель {MODEL_PATH} не найдена!")
            return False
        
        print(f"Найдена модель {MODEL_PATH}, размер: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} МБ")
        
        # Ищем тестовые изображения
        test_images = []
        search_paths = [
            os.path.join(DATA_DIR, 'test/*/*.png'),
            os.path.join(DATA_DIR, 'valid/valid_images/*.png'),
            os.path.join(DATA_DIR, 'train/train_images/*.png')
        ]
        
        for pattern in search_paths:
            found_images = glob.glob(pattern)
            if found_images:
                test_images.extend(found_images[:5])  # Берем до 5 изображений из каждой папки
                if len(test_images) >= 5:
                    break
                
        if not test_images:
            print("Тестовые изображения не найдены!")
            return False
            
        print(f"Найдено {len(test_images)} тестовых изображений")
        
        # Загружаем модель
        print(f"Загружаем модель {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        print("Модель загружена успешно!")
        
        # Запускаем инференс на изображениях с низким порогом обнаружения
        print(f"Запускаем инференс с порогом обнаружения {CONF_THRESHOLD}...")
        
        for img_path in test_images:
            print(f"Обработка: {img_path}")
            
            try:
                results = model.predict(
                    source=img_path,
                    conf=CONF_THRESHOLD,  # Очень низкий порог уверенности
                    iou=0.25,  # Менее строгий порог IoU
                    save=True,
                    project=OUTPUT_DIR,
                    name='results',
                    line_width=1,  # Тонкие линии для визуализации
                    show_labels=True,  # Показывать метки
                    show_conf=True,   # Показывать значение уверенности
                    verbose=True      # Подробный вывод
                )
                
                # Выводим информацию о найденных объектах
                for i, result in enumerate(results):
                    boxes = result.boxes
                    masks = result.masks
                    
                    if boxes is not None:
                        print(f"  - Обнаружено объектов (bounding boxes): {len(boxes)}")
                    
                    if masks is not None:
                        print(f"  - Обнаружено масок: {len(masks)}")
                    else:
                        print("  - Маски не обнаружены")
                
                print(f"  - Результаты сохранены: {results[0].save_dir}")
                
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
        
        print("Инференс завершен успешно!")
        return True
        
    except Exception as e:
        print(f"Ошибка при инференсе: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    predict_on_images() 