import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
import sys
import glob
import time
import matplotlib.pyplot as plt
import shutil
import random

# Заголовок приложения
st.title("Обнаружение пузырьков на видео")
st.write("Загрузите видео для обработки с помощью модели YOLO")

# Проверяем наличие PyTorch и CUDA
try:
    import torch
    from ultralytics import YOLO
    
    # Проверка доступности GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.info(f"Используется устройство: {device}")
    
    # Загрузка модели
    @st.cache_resource
    def load_model(model_path):
        if not os.path.exists(model_path):
            st.error(f"Модель {model_path} не найдена!")
            return None
        st.info(f"Загрузка модели из: {model_path}")
        model = YOLO(model_path)
        
        # Использование half precision вызывает ошибку типов данных
        # Закомментируем эту часть, чтобы избежать ошибки
        """
        if device == 'cuda' and torch.cuda.is_available():
            st.info("Используется half precision (FP16) для ускорения")
            model.to('cuda')
            model.model.half()
        """
        
        return model
    
    # Путь к модели
    model_dir = r"C:\Users\xdddd\FrothModel\yolo_runs\bubbles_segmentation_memory_optimized\weights"
    model_path = os.path.join(model_dir, "best.pt")
    
    # Проверка наличия файла модели
    if not os.path.exists(model_path):
        st.error(f"Модель не найдена по пути: {model_path}")
        # Попытка найти любой .pt файл в указанной директории
        model_files = glob.glob(os.path.join(model_dir, "*.pt"))
        if model_files:
            model_path = model_files[0]
            st.info(f"Найдена альтернативная модель: {model_path}")
        else:
            st.error("Не найдено ни одной модели в указанной директории")
            st.stop()
    
    try:
        model = load_model(model_path)
        if model:
            st.success(f"Модель успешно загружена")
        else:
            st.error("Не удалось загрузить модель")
            st.stop()
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.stop()
    
except ImportError as e:
    st.error(f"Ошибка импорта модулей: {e}")
    st.error("Убедитесь, что установлены все необходимые библиотеки")
    st.stop()

# Настройки обнаружения
confidence = st.slider("Порог уверенности", min_value=0.01, max_value=1.0, value=0.01, step=0.01)
iou = st.slider("Порог IoU", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

# Настройки отображения
show_only_masks = st.checkbox("Показывать только маски без боксов", value=True)
show_confidence = st.checkbox("Показывать значения уверенности", value=False)
mask_opacity = st.slider("Прозрачность масок", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

# Настройки оптимизации
st.subheader("Настройки оптимизации скорости")
with st.expander("Параметры оптимизации инференса"):
    resize_factor = st.slider(
        "Масштаб для входных кадров (меньше = быстрее)", 
        min_value=0.25, 
        max_value=1.0, 
        value=1.0, 
        step=0.05,
        help="Уменьшение размера входных кадров для ускорения. 1.0 = оригинальный размер, 0.5 = половина размера и т.д."
    )
    
    process_every_nth_frame = st.slider(
        "Обрабатывать каждый N-й кадр", 
        min_value=1, 
        max_value=10, 
        value=1, 
        step=1,
        help="Пропуск кадров для ускорения обработки. Значение 1 обрабатывает каждый кадр, 2 - каждый второй и т.д."
    )
    
    save_frames = st.checkbox(
        "Сохранять все обработанные кадры", 
        value=True,
        help="Отключение сохранения каждого кадра может ускорить обработку, но вы не сможете скачать отдельные кадры"
    )
    
    cuda_clear_frequency = st.slider(
        "Частота очистки CUDA памяти (каждые N кадров)", 
        min_value=10, 
        max_value=100, 
        value=30, 
        step=10,
        help="Более редкая очистка памяти может ускорить обработку, но может привести к утечкам памяти на длинных видео"
    )

# Функция для отображения графика FPS
def plot_fps_chart(fps_values):
    if not fps_values:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fps_values)
    ax.set_title('Производительность инференса (FPS)')
    ax.set_xlabel('Кадр')
    ax.set_ylabel('FPS')
    ax.grid(True)
    
    return fig

# Функция для создания превью результатов
def create_preview_grid(frames, rows=2, cols=2):
    if not frames:
        return None
    
    # Выбираем несколько кадров для превью
    n_frames = len(frames)
    step = max(1, n_frames // (rows * cols))
    selected_frames = [frames[i] for i in range(0, n_frames, step)][:rows*cols]
    
    if not selected_frames:
        return None
    
    # Размер одного кадра
    h, w, _ = selected_frames[0].shape
    
    # Создаем сетку изображений
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    for i, frame in enumerate(selected_frames):
        r, c = i // cols, i % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = frame
    
    return grid

# Функция для отрисовки только масок сегментации
def draw_masks_only(image, results, opacity=0.5):
    # Создаем копию исходного изображения
    output_image = image.copy()
    
    # Проверяем наличие масок
    if results.masks is None:
        return output_image
    
    # Генерируем случайные цвета для каждой маски, если их больше 20
    num_masks = len(results.masks)
    
    # Создаем палитру цветов
    if num_masks > 20:
        colors = []
        for _ in range(num_masks):
            colors.append((
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ))
    else:
        # Используем предопределенные цвета для небольшого количества масок
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0),
            (192, 0, 192), (0, 192, 192), (64, 0, 0), (0, 64, 0)
        ]
    
    # Получаем размеры изображения
    h, w = output_image.shape[:2]
    
    # Создаем общую маску для всех объектов
    combined_mask = np.zeros_like(output_image)
    
    # Наносим каждую маску на изображение
    for i, mask in enumerate(results.masks.data):
        # Преобразуем маску в формат uint8 для OpenCV
        binary_mask = mask.cpu().numpy().astype(np.uint8)
        
        # Изменяем размер маски, чтобы она соответствовала размеру изображения
        if binary_mask.shape[:2] != (h, w):
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Создаем цветную маску только для областей с значением 1
        color = colors[i % len(colors)]
        
        # Применяем цвет только к ненулевым пикселям маски
        combined_mask[binary_mask > 0] = color
    
    # Накладываем маску на изображение с заданной прозрачностью
    # Сначала создаем промежуточную цветную маску
    color_overlay = np.zeros_like(output_image)
    
    # Копируем комбинированную маску
    color_overlay = combined_mask.copy()
    
    # Накладываем комбинированную цветную маску на исходное изображение
    cv2.addWeighted(color_overlay, opacity, output_image, 1.0, 0, output_image)
    
    return output_image

# Функция для изменения размера изображения с сохранением соотношения сторон
def resize_with_aspect_ratio(image, scale_factor):
    if scale_factor == 1.0:
        return image
    
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Функция для обработки видео и сохранения результата
def process_video(video_path, confidence, iou, show_only_masks, show_confidence, mask_opacity, 
                  resize_factor=1.0, process_every_nth_frame=1, save_frames=True, cuda_clear_frequency=30):
    # Получение информации о видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Не удалось открыть видео файл")
        return None, None, 0, None, None
            
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Создаем директорию для кадров если нужно сохранять кадры
    frames_dir = None
    if save_frames:
        frames_dir = os.path.join(tempfile.gettempdir(), "processed_frames")
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
    
    # Создаем файл для выходного видео
    output_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    
    # Пробуем различные кодеки для совместимости с Windows
    codecs_to_try = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('DIVX', '.avi'),
        ('H264', '.mp4')
    ]
    
    output = None
    for codec, ext in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_path = os.path.join(tempfile.gettempdir(), f"processed_video{ext}")
        
        output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if output.isOpened():
            output_video_path = output_path
            break
    
    if not output or not output.isOpened():
        st.error("Не удалось создать файл для выходного видео. Будут сохранены только кадры.")
    
    # Статистика инференса
    frame_count = 0
    processed_frame_count = 0
    detected_objects = 0
    detected_objects_per_frame = []
    processing_times = []
    fps_values = []
    preview_frames = []
    start_time_total = time.time()
    
    # Создаем элементы для отображения прогресса
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Статистика инференса
    inference_stats = st.empty()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Пропускаем кадры если нужно
            if frame_count % process_every_nth_frame != 0:
                frame_count += 1
                continue
            
            # Изменяем размер кадра для ускорения инференса
            if resize_factor != 1.0:
                processing_frame = resize_with_aspect_ratio(frame, resize_factor)
            else:
                processing_frame = frame
            
            # Замеряем время обработки кадра
            start_time = time.time()
            
            # Обработка кадра с YOLO
            results = model.predict(
                source=processing_frame,
                conf=confidence,
                iou=iou,
                verbose=False
            )
            
            # Замеряем время окончания обработки
            inference_time = time.time() - start_time
            processing_times.append(inference_time)
            current_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_values.append(current_fps)
            
            # Подсчет обнаруженных объектов в текущем кадре
            objects_in_frame = 0
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                objects_in_frame = len(results[0].masks)
            elif hasattr(results[0], 'boxes') and results[0].boxes is not None:
                objects_in_frame = len(results[0].boxes)
            
            detected_objects += objects_in_frame
            detected_objects_per_frame.append(objects_in_frame)
            
            # Получение аннотированного кадра - работаем с оригинальным размером
            if show_only_masks:
                # Отображаем только маски без боксов
                annotated_frame = draw_masks_only(frame, results[0], opacity=mask_opacity)
            else:
                # Если размер изменяли, нужно применить результаты к оригинальному кадру
                if resize_factor != 1.0:
                    # Используем стандартную визуализацию YOLO с применением к оригинальному кадру
                    annotated_frame = frame.copy()
                    # Масштабируем результаты обратно к оригинальному размеру
                    results[0].plot(annotated_frame, line_width=1, show_labels=not show_only_masks, show_conf=show_confidence)
                else:
                    # Используем стандартную визуализацию YOLO
                    annotated_frame = results[0].plot(
                        show_labels=not show_only_masks,
                        show_conf=show_confidence,
                        line_width=1
                    )
            
            # Сохраняем кадр для создания превью
            if frame_count % max(1, total_frames // 10) == 0:  # Сохраняем ~10 кадров для превью
                preview_frames.append(annotated_frame)
            
            # Сохраняем каждый обработанный кадр, если включена опция
            if save_frames and frames_dir:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg"), annotated_frame)
            
            # Запись в видеофайл
            if output and output.isOpened():
                output.write(annotated_frame)
            
            # Обновление прогресса
            frame_count += 1
            processed_frame_count += 1
            progress = int(frame_count / total_frames * 100)
            progress_text.text(f"Обработано кадров: {frame_count}/{total_frames} | Обнаружено объектов: {objects_in_frame}")
            progress_bar.progress(progress)
            
            # Обновление статистики инференса
            avg_fps = processed_frame_count / (time.time() - start_time_total)
            avg_inference = sum(processing_times) / len(processing_times) * 1000  # мс
            inference_stats.text(f"Инференс: {avg_inference:.1f} мс/кадр | FPS: {avg_fps:.1f} | Текущий FPS: {current_fps:.1f}")
            
            # Очищаем память CUDA с указанной частотой
            if device == 'cuda' and frame_count % cuda_clear_frequency == 0:
                torch.cuda.empty_cache()
        
        # Закрываем ресурсы
        if output and output.isOpened():
            output.release()
        cap.release()
        
        # Финальная статистика
        total_time = time.time() - start_time_total
        progress_text.text(f"Обработано кадров: {frame_count}/{total_frames} (Завершено)")
        
        # Создаем сетку превью из обработанных кадров
        preview_grid = None
        if preview_frames:
            preview_grid = create_preview_grid(preview_frames)
            if preview_grid is not None:
                preview_path = os.path.join(tempfile.gettempdir(), "preview_grid.jpg")
                cv2.imwrite(preview_path, preview_grid)
        
        # Если видео не сохранено, попробуем создать его из кадров
        if save_frames and (not output or not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0) and os.path.exists(frames_dir):
            st.warning("Создание видео с помощью OpenCV не удалось. Попытка создать видео из кадров...")
            
            # Создаем видео из кадров с помощью ffmpeg, если он доступен
            try:
                # Используем ffmpeg для создания видео из кадров
                import subprocess
                output_video_path = os.path.join(tempfile.gettempdir(), "processed_video_from_frames.mp4")
                
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(frames_dir, "frame_%06d.jpg"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    output_video_path
                ]
                
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                    st.success("Видео успешно создано с помощью ffmpeg!")
                else:
                    st.error("Не удалось создать видео с помощью ffmpeg.")
            except Exception as e:
                st.error(f"Ошибка при создании видео через ffmpeg: {e}")
        
        # Если всё в порядке, возвращаем статистику и пути к результатам
        inference_statistics = {
            'total_frames': frame_count,
            'processed_frames': processed_frame_count,
            'total_time': total_time,
            'avg_time_per_frame': sum(processing_times) / len(processing_times) if processing_times else 0,
            'min_time': min(processing_times) if processing_times else 0,
            'max_time': max(processing_times) if processing_times else 0,
            'fps_values': fps_values,
            'avg_fps': processed_frame_count / total_time if total_time > 0 else 0,
            'objects_per_frame': detected_objects_per_frame,
            'avg_objects_per_frame': sum(detected_objects_per_frame) / len(detected_objects_per_frame) if detected_objects_per_frame else 0,
            'max_objects_in_frame': max(detected_objects_per_frame) if detected_objects_per_frame else 0
        }
        
        return frames_dir, output_video_path, detected_objects, inference_statistics, preview_frames
        
    except Exception as e:
        st.error(f"Ошибка при обработке видео: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, 0, None, None
    finally:
        # Закрываем ресурсы
        if cap.isOpened():
            cap.release()
        if 'output' in locals() and output and output.isOpened():
            output.release()

# Функция для построения графика количества объектов на кадр
def plot_objects_per_frame(objects_per_frame):
    if not objects_per_frame:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(objects_per_frame)
    ax.set_title('Количество обнаруженных пузырьков на кадр')
    ax.set_xlabel('Кадр')
    ax.set_ylabel('Количество пузырьков')
    ax.grid(True)
    
    return fig

# Загрузка видео
uploaded_video = st.file_uploader("Выберите видео файл", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    try:
        # Сохранение загруженного видео во временный файл
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name
        temp_file.close()
        
        # Отображение исходного видео
        st.video(video_path)
        
        # Обработка видео
        if st.button("Обработать видео"):
            with st.spinner("Обработка видео..."):
                frames_dir, output_video_path, detected_objects, inference_stats, preview_frames = process_video(
                    video_path, confidence, iou, show_only_masks, show_confidence, mask_opacity,
                    resize_factor, process_every_nth_frame, save_frames, cuda_clear_frequency
                )
                
                if (frames_dir and os.path.exists(frames_dir)) or output_video_path:
                    # Успешная обработка
                    st.success("Обработка видео завершена!")
                    
                    # Отображение обработанного видео
                    if output_video_path and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                        st.subheader("Обработанное видео")
                        st.video(output_video_path)
                        
                        # Добавляем кнопку скачивания видео
                        with open(output_video_path, "rb") as f:
                            st.download_button(
                                label="Скачать обработанное видео",
                                data=f,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.warning("Видео не удалось создать или отобразить.")
                    
                    # Отображение превью результатов
                    if preview_frames:
                        st.subheader("Обработанные кадры (превью)")
                        
                        # Конвертируем кадры в формат RGB для отображения в Streamlit
                        cols = st.columns(2)
                        for i, frame in enumerate(preview_frames[:4]):  # Показываем до 4 кадров
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            cols[i % 2].image(rgb_frame, caption=f"Кадр {i+1}", use_column_width=True)
                    
                    # Отображение статистики инференса
                    if inference_stats:
                        st.subheader("Статистика инференса")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Всего кадров в видео", inference_stats['total_frames'])
                            st.metric("Обработано кадров", inference_stats['processed_frames'])
                            st.metric("Обнаружено объектов", detected_objects)
                            st.metric("Среднее число объектов на кадр", f"{inference_stats['avg_objects_per_frame']:.1f}")
                        
                        with col2:
                            st.metric("Общее время обработки", f"{inference_stats['total_time']:.1f} сек")
                            st.metric("Среднее время инференса", f"{inference_stats['avg_time_per_frame']*1000:.1f} мс")
                            st.metric("Средний FPS", f"{inference_stats['avg_fps']:.1f}")
                            st.metric("Макс. объектов в кадре", inference_stats['max_objects_in_frame'])
                        
                        # График FPS
                        st.subheader("График производительности")
                        fps_chart = plot_fps_chart(inference_stats['fps_values'])
                        if fps_chart:
                            st.pyplot(fps_chart)
                        
                        # График числа объектов на кадр
                        st.subheader("График количества пузырьков")
                        objects_chart = plot_objects_per_frame(inference_stats['objects_per_frame'])
                        if objects_chart:
                            st.pyplot(objects_chart)
                    
                    # Информация о настройках
                    st.subheader("Использованные настройки")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Порог уверенности: {confidence}")
                        st.write(f"Порог IoU: {iou}")
                        st.write(f"Режим отображения: {'Только маски' if show_only_masks else 'Маски и боксы'}")
                    
                    with col2:
                        st.write(f"Масштаб кадров: {resize_factor:.2f}x")
                        st.write(f"Обработка каждого {process_every_nth_frame}-го кадра")
                        st.write(f"Сохранение кадров: {save_frames}")
                    
                    # Предлагаем скачать обработанные кадры если они были сохранены
                    if save_frames and frames_dir and os.path.exists(frames_dir):
                        zip_path = os.path.join(tempfile.gettempdir(), "processed_frames.zip")
                        shutil.make_archive(zip_path[:-4], 'zip', frames_dir)
                        
                        # Добавляем кнопку скачивания кадров
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label="Скачать обработанные кадры",
                                data=f,
                                file_name="processed_frames.zip",
                                mime="application/zip"
                            )
                    
                else:
                    st.error("Ошибка при обработке видео")
                    
    except Exception as e:
        st.error(f"Ошибка: {e}")
        import traceback
        st.error(traceback.format_exc())
    finally:
        # Удаляем временный файл
        if 'video_path' in locals() and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except:
                pass

# Информация по использованию
st.markdown("""
## Инструкция по использованию
1. Загрузите видео файл в формате MP4, AVI или MOV
2. Настройте параметры обнаружения с помощью ползунков
   - Низкий порог уверенности позволяет обнаружить больше пузырьков
   - Включите "Показывать только маски" для отображения сегментации без боксов
3. Настройте прозрачность масок для лучшей визуализации
4. **Для ускорения обработки**:
   - Уменьшите масштаб кадров (до 0.25-0.5 для существенного ускорения)
   - Увеличьте шаг пропуска кадров (обрабатывать каждый 2-3 кадр)
   - Отключите сохранение всех кадров, если вам нужно только видео
5. Нажмите кнопку "Обработать видео"
6. Дождитесь завершения обработки
7. Просмотрите обработанное видео с масками, статистику инференса и скачайте результаты при необходимости
""") 