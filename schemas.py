from pydantic import BaseModel
from typing import List, Dict, Optional

class DetectedObject(BaseModel):
    label: str
    confidence: float
    box: List[int] # [xmin, ymin, xmax, ymax]

class FrameAnalysisResult(BaseModel):
    frame_index: int
    timestamp_sec: float
    detected_date: Optional[str] = None
    task_name: Optional[str] = None
    target_objects: List[str] = []
    found_counts: Dict[str, int] = {}
    is_completed: bool = False
    annotated_image_base64: Optional[str] = None # Картинка с нарисованными рамками
    detected_objects: List[DetectedObject] = []

class VideoAnalysisResponse(BaseModel):
    video_name: str
    total_frames_analyzed: int
    overall_verdict: bool
    details: List[FrameAnalysisResult]