import cv2
import base64
import numpy as np
from datetime import datetime
from typing import List, Tuple
from schemas import DetectedObject, FrameAnalysisResult
from services.excel_service import excel_service

class MLService:
    def __init__(self):
        # self.yolo = YOLO("yolov8x-worldv2.pt")
        print("ML Service инициализирован.")

    def extract_date_from_frame(self, frame: np.ndarray) -> datetime:
        # TODO: Напарница вставит сюда легкий OCR (напр. easyocr или кроп таймштампа)
        # Пока возвращаем текущую дату как заглушку:
        return datetime.now()

    def detect_target_objects(self, frame: np.ndarray, target_classes: List[str]) -> List[DetectedObject]:
        # TODO: Заменить на self.yolo.set_classes(target_classes) и predict()
        # Заглушка: если ищем кота, делаем вид, что нашли
        detections = []
        if "cat" in target_classes:
            detections.append(
                DetectedObject(label="cat", confidence=0.92, box=[50, 50, 300, 300])
            )
        return detections

    def draw_boxes_and_encode(self, frame: np.ndarray, detections: List[DetectedObject]) -> str:
        """Рисует прямоугольники поверх кадра и превращает его в base64 строку"""
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            # Рисуем зеленый прямоугольник
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Пишем текст
            label = f"{det.label} {det.confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Конвертируем обратно в JPEG и затем в base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"

    def process_single_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> FrameAnalysisResult:
        # Шаг 1: Распознаем дату на кадре
        detected_dt = self.extract_date_from_frame(frame)
        date_str = detected_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Шаг 2 и 3: Идем в Excel за задачей и маппингом
        task_name, target_objects = excel_service.get_task_and_targets(detected_dt)

        # Шаг 4: Детектим через YOLO только нужные целевые объекты
        detections = self.detect_target_objects(frame, target_objects)

        # Подсчет количества найденных объектов
        counts = {}
        for det in detections:
            counts[det.label] = counts.get(det.label, 0) + 1

        # Вердикт: выполнена ли задача (например, найдены ли все требуемые объекты)
        is_completed = len(detections) > 0 and all(obj in counts for obj in target_objects)

        # Шаг 5: Отрисовываем боксы
        base64_img = self.draw_boxes_and_encode(frame, detections)

        return FrameAnalysisResult(
            frame_index=frame_idx,
            timestamp_sec=timestamp,
            detected_date=date_str,
            task_name=task_name,
            target_objects=target_objects,
            found_counts=counts,
            is_completed=is_completed,
            annotated_image_base64=base64_img,
            detected_objects=detections
        )

ml_service = MLService()