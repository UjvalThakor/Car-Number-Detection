import cv2
import easyocr
import numpy as np
import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class DetectedPlate:
    plate_text: str
    confidence: float
    frame_number: int
    bounding_box: Optional[tuple] = None

    def to_dict(self):
        return {
            "plate_text": self.plate_text,
            "confidence": round(self.confidence, 4),
            "frame_number": self.frame_number,
            "bounding_box": self.bounding_box,
        }

class NumberPlateDetector:

    BLACKLIST = {
        "SUBSCRIBE", "CAMERA", "NUMBERPLATE", "FORGET", "LIKE", "SHARE",
        "FOLLOW", "CLICK", "COMMENT", "WATCH", "VIDEO", "YOUTUBE", "CHANNEL"
    }

    def __init__(
            self,
            frame_skip: int = 10,
            confidence_threshold: float = 0.3,
            gpu: bool = False,
            languages: list = None,
    ):
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.languages = languages or ["en"]

        logger.info("Loading EasyOCR model.....")
        self.reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)
        logger.info("EasyOCR Ready :)")

    def process_video(self, video_path: str) -> list:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_plates: list = []
        frame_number = 0

        # Process 1 frame per second
        self.frame_skip = max(1, int(fps))

        logger.info(f"Video: {total_frames} frames @ {fps:.1f}fps — scanning 1 frame/sec")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % self.frame_skip == 0:
                    small_frame = self._resize_frame(frame)
                    plates = self._detect_in_frame(small_frame, frame_number)
                    all_plates.extend(plates)

                    if plates:
                        logger.info(f"Frame {frame_number}: found {[p.plate_text for p in plates]}")

                frame_number += 1

        finally:
            cap.release()

        unique_plates = self._deduplicate(all_plates)
        logger.info(f"Done. Unique plates: {len(unique_plates)}")
        return [p.to_dict() for p in unique_plates]

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return frame

    def _detect_in_frame(self, frame: np.ndarray, frame_number: int) -> list:
        plates = []

        # Method 1: Region-based detection
        regions = self._find_plate_regions(frame)
        for (x, y, w, h) in regions:
            crop = frame[y:y+h, x:x+w]
            processed = self._preprocess(crop)
            results = self.reader.readtext(processed)
            for (_, text, confidence) in results:
                if confidence < self.confidence_threshold:
                    continue
                cleaned = self._clean_text(text)
                if cleaned:
                    plates.append(DetectedPlate(
                        plate_text=cleaned,
                        confidence=confidence,
                        frame_number=frame_number,
                        bounding_box=(x, y, w, h)
                    ))

        if not plates:
            results = self.reader.readtext(frame)
            for (bbox, text, confidence) in results:
                if confidence < self.confidence_threshold:
                    continue
                cleaned = self._clean_text(text)
                if cleaned:
                    pts = np.array(bbox, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    plates.append(DetectedPlate(
                        plate_text=cleaned,
                        confidence=confidence,
                        frame_number=frame_number,
                        bounding_box=(x, y, w, h)
                    ))

        return plates

    def _find_plate_regions(self, frame: np.ndarray) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]
        regions = []
        seen = set()

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h) if h > 0 else 0
            area = w * h

            if 1.5 <= aspect <= 7.0 and 0.0003 * frame_area <= area <= 0.15 * frame_area:
                key = (x // 10, y // 10)
                if key not in seen:
                    seen.add(key)
                    regions.append((x, y, w, h))

        return regions

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return img

        scale = 80 / h
        img = cv2.resize(img, (max(1, int(w * scale)), 80), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper().strip())
        has_letter = bool(re.search(r"[A-Z]", cleaned))
        has_digit = bool(re.search(r"[0-9]", cleaned))
        if not (has_letter and has_digit and 4 <= len(cleaned) <= 10):
            return ""
        if cleaned in self.BLACKLIST:
            return ""
        return cleaned

    def _deduplicate(self, plates: list) -> list:
        best = {}
        for p in plates:
            if p.plate_text not in best or p.confidence > best[p.plate_text].confidence:
                best[p.plate_text] = p
        return list(best.values())