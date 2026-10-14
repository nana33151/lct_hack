import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import VideoAnalysisResponse
from services.video_service import video_processor
from services.ml_service import ml_service

app = FastAPI(title="Video Audit System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    # Валидация расширения
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Формат файла должен быть видео (.mp4, .avi, etc.)")

    # Сохраняем входящий поток во временный файл на диск
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_video_path = tmp_file.name

    frame_results = []
    
    try:
        # Нарезаем видео (1 кадр в секунду, чтобы не греть сервер)
        for frame_idx, timestamp, frame in video_processor.extract_frames(tmp_video_path, step_fps=1):
            # Прогоняем весь пайплайн по кадру
            res = ml_service.process_single_frame(frame, frame_idx, timestamp)
            frame_results.append(res)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки видео: {str(e)}")
    finally:
        # Обязательно удаляем временное видео после завершения
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)

    # Общий вердикт: например, успешно, если хоть на половине кадров задача выполнена
    completed_frames = sum(1 for r in frame_results if r.is_completed)
    overall = (completed_frames / len(frame_results) > 0.5) if frame_results else False

    return VideoAnalysisResponse(
        video_name=file.filename,
        total_frames_analyzed=len(frame_results),
        overall_verdict=overall,
        details=frame_results
    )