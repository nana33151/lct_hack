import cv2
from typing import Generator, Tuple
import numpy as np

class VideoProcessor:
    @staticmethod
    def extract_frames(video_path: str, step_fps: int = 1) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Генератор: отдает (номер_кадра, таймкод_в_секундах, сам_кадр_numpy)
        step_fps=1 означает, что берем 1 кадр в секунду
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть видеофайл")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = int(video_fps / step_fps)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps
                yield frame_idx, timestamp, frame

            frame_idx += 1

        cap.release()

video_processor = VideoProcessor()