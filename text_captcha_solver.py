
"""
Text CAPTCHA solver that applies advanced preprocessing and character segmentation
before running OCR (or a CRNN). Returns concatenated text prediction.
"""
import logging
import time
from typing import Optional, List
import numpy as np
import cv2
from captcha_solver.preprocessing.image_preprecessor import preprocess_image, segment_characters
from captcha_solver.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextCaptchaSolver:
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.timeout = self.config.get("timeout", 30)

    def solve(self, image_path: str) -> Optional[str]:
        """
        Full pipeline: preprocess -> segment -> predict each char -> assemble -> threshold
        """
        start = time.time()
        try:
            logger.info(f"Preprocessing image: {image_path}")
            binary = preprocess_image(image_path)
            chars = segment_characters(binary)
            if not chars:
                logger.warning("No characters segmented from image")
                return None
            logger.info(f"Segmented {len(chars)} character regions")

            result = []
            confidences: List[float] = []
            for idx, char_img in enumerate(chars):
                # Model expects numpy array or PIL; here pass numpy
                pred = self.model.predict(char_img)
                if pred.confidence < self.confidence_threshold:
                    logger.warning(f"Char {idx} low confidence: {pred.text} ({pred.confidence:.2f})")
                    return None
                result.append(pred.text)
                confidences.append(pred.confidence)

            text = "".join(result)
            avg_conf = float(np.mean(confidences))
            elapsed = time.time() - start
            logger.info(f"Solved CAPTCHA '{text}' (avg_conf={avg_conf:.2f}) in {elapsed:.2f}s")
            return text

        except Exception as e:
            logger.exception(f"Error in TextCaptchaSolver: {e}")
            return None
