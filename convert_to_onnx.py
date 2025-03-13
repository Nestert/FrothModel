import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import argparse
import onnx
import onnxruntime as ort
from torchvision import transforms

# Константы (должны соответствовать training.py)
IMAGE_SIZE = (512, 512)
MOBILENETV3_SIZE = 'large'
MOBILENETV3_WEIGHTS = 'imagenet'

def load_pytorch_model(model_path, device):
    """
    Загружает предварительно обученную PyTorch модель
    """
    print(f"Загрузка модели из {model_path}...")
    
    # Создаем модель с такой же архитектурой, как при обучении
    model = smp.Unet(
        encoder_name=f"timm-mobilenetv3_{MOBILENETV3_SIZE}_100",
        encoder_weights=None,  # Веса будут загружены из сохраненной модели
        in_channels=3,
        classes=1,
        activation=None  # Без активации, будем использовать сигмоиду при необходимости
    )
    
    # Загружаем веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Переводим модель в режим оценки
    
    return model

def convert_to_onnx(model, device, onnx_path, input_shape=(1, 3, 512, 512), dynamic_batch=True):
    """
    Конвертирует PyTorch модель в формат ONNX
    """
    print(f"Преобразование модели в формат ONNX ({onnx_path})...")
    
    # Создаем тестовый входной тензор
    dummy_input = torch.randn(input_shape, device=device)
    
    # Динамические оси для поддержки переменного размера батча
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Экспортируем модель в ONNX
    torch.onnx.export(
        model,                     # PyTorch модель
        dummy_input,               # Пример входных данных
        onnx_path,                 # Путь для сохранения
        export_params=True,        # Сохраняем веса модели
        opset_version=12,          # Версия ONNX операторов
        do_constant_folding=True,  # Оптимизация констант
        input_names=['input'],     # Имя входного тензора
        output_names=['output'],   # Имя выходного тензора
        dynamic_axes=dynamic_axes   # Динамические размеры для батча
    )
    
    print(f"Модель успешно преобразована и сохранена в {onnx_path}")
    
    return onnx_path

def validate_onnx_model(onnx_path):
    """
    Проверяет корректность ONNX модели
    """
    print(f"Проверка ONNX модели ({onnx_path})...")
    
    try:
        # Загружаем ONNX модель
        onnx_model = onnx.load(onnx_path)
        
        # Проверяем модель на ошибки
        onnx.checker.check_model(onnx_model)
        
        print("ONNX модель проверена, структура корректна.")
        return True
    except Exception as e:
        print(f"Ошибка при проверке ONNX модели: {e}")
        return False

def optimize_onnx_model(input_onnx_path, output_onnx_path):
    """
    Оптимизирует ONNX модель с помощью ONNX Runtime
    """
    print(f"Оптимизация ONNX модели...")
    
    try:
        from onnxruntime.transformers import optimizer
        
        # Путь к оптимизированной модели
        optimized_model = optimizer.optimize_model(
            input_onnx_path,
            model_type='conv',  # Для CNN моделей
            num_heads=0,        # Не используется для CNN
            hidden_size=0       # Не используется для CNN
        )
        
        # Сохраняем оптимизированную модель
        optimized_model.save_model_to_file(output_onnx_path)
        
        print(f"ONNX модель успешно оптимизирована и сохранена в {output_onnx_path}")
        return True
    except Exception as e:
        print(f"Предупреждение: не удалось оптимизировать ONNX модель: {e}")
        print("Продолжаем без оптимизации...")
        return False

def compare_outputs(pytorch_model, onnx_path, device, test_image_path=None):
    """
    Сравнивает выходы PyTorch и ONNX моделей на одном и том же входе
    """
    print("Сравнение выходов PyTorch и ONNX моделей...")
    
    # Если тестовое изображение не указано, используем случайный тензор
    if test_image_path and os.path.exists(test_image_path):
        print(f"Используем тестовое изображение: {test_image_path}")
        # Загружаем и подготавливаем изображение
        img = Image.open(test_image_path).convert("RGB")
        # Изменяем размер
        img = img.resize(IMAGE_SIZE)
        # Преобразуем в тензор
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)
    else:
        print("Используем случайный тензор для сравнения выходов...")
        input_tensor = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)
    
    # Получаем выход PyTorch модели
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)
        pytorch_output = torch.sigmoid(pytorch_output).cpu().numpy()
    
    # Получаем выход ONNX модели
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    # Преобразуем входной тензор для ONNX Runtime
    onnx_input = input_tensor.cpu().numpy()
    
    # Запускаем ONNX Runtime
    onnx_output = ort_session.run([output_name], {input_name: onnx_input})[0]
    
    # Применяем сигмоиду к выходу ONNX
    onnx_output = 1 / (1 + np.exp(-onnx_output))
    
    # Сравниваем выходы
    output_diff = np.abs(pytorch_output - onnx_output).mean()
    print(f"Среднее абсолютное различие между выходами: {output_diff}")
    
    if output_diff < 1e-4:
        print("Выходы PyTorch и ONNX моделей совпадают с высокой точностью!")
        return True
    else:
        print("Предупреждение: заметны различия между выходами PyTorch и ONNX моделей.")
        print(f"PyTorch выход (мин/макс/средн): {pytorch_output.min()}/{pytorch_output.max()}/{pytorch_output.mean()}")
        print(f"ONNX выход (мин/макс/средн): {onnx_output.min()}/{onnx_output.max()}/{onnx_output.mean()}")
        return False

def benchmark_onnx_model(onnx_path, input_shape=(1, 3, 512, 512), num_iterations=100):
    """
    Тестирует производительность ONNX модели
    """
    print(f"Тестирование производительности ONNX модели ({onnx_path})...")
    
    # Создаем сессию ONNX Runtime
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    # Создаем случайный входной тензор
    random_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Прогреваем модель
    for _ in range(10):
        session.run(None, {input_name: random_input})
    
    # Измеряем время выполнения
    import time
    start_time = time.time()
    
    for _ in range(num_iterations):
        session.run(None, {input_name: random_input})
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations * 1000  # в миллисекундах
    
    print(f"Среднее время выполнения: {avg_time:.2f} мс (для батча размером {input_shape[0]})")
    print(f"FPS: {1000 / avg_time * input_shape[0]:.2f}")
    
    return avg_time

def main():
    parser = argparse.ArgumentParser(description='Конвертация PyTorch модели в ONNX формат')
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='Путь к файлу PyTorch модели (.pth)')
    parser.add_argument('--output', type=str, default='model.onnx',
                        help='Путь к выходному ONNX файлу')
    parser.add_argument('--optimized_output', type=str, default='model_optimized.onnx',
                        help='Путь к оптимизированному ONNX файлу')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Путь к тестовому изображению для сравнения выходов')
    parser.add_argument('--benchmark', action='store_true',
                        help='Выполнить тестирование производительности')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Размер батча для экспорта модели')
    
    args = parser.parse_args()
    
    # Проверяем доступность устройства CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Проверяем наличие файла модели
    if not os.path.exists(args.model):
        print(f"Ошибка: файл модели '{args.model}' не найден!")
        return
    
    # Загружаем модель PyTorch
    model = load_pytorch_model(args.model, device)
    
    # Конвертируем в ONNX
    input_shape = (args.batch_size, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    onnx_path = convert_to_onnx(model, device, args.output, input_shape=input_shape)
    
    # Проверяем корректность ONNX модели
    if validate_onnx_model(onnx_path):
        # Оптимизируем ONNX модель
        try:
            optimize_onnx_model(onnx_path, args.optimized_output)
            optimized_onnx_path = args.optimized_output
        except Exception:
            print("Не удалось оптимизировать модель, будем использовать неоптимизированную версию.")
            optimized_onnx_path = onnx_path
        
        # Сравниваем выходы PyTorch и ONNX моделей
        compare_outputs(model, onnx_path, device, args.test_image)
        
        # Тестируем производительность ONNX модели, если указан флаг --benchmark
        if args.benchmark and os.path.exists(optimized_onnx_path):
            benchmark_onnx_model(optimized_onnx_path, input_shape=input_shape)
            
            # Если есть оптимизированная версия, сравниваем с ней тоже
            if optimized_onnx_path != onnx_path:
                benchmark_onnx_model(onnx_path, input_shape=input_shape)
    
    print("Конвертация завершена!")

if __name__ == '__main__':
    main() 