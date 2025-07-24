import logging
import time
from typing import Optional
import cv2
import numpy as np
import pytesseract
from captcha_solver.preprocessing.image_preprecessor import preprocess_image
from captcha_solver.utils.logger import setup_logger

logger = setup_logger(__name__)

class TesseractSolver:
    """
    Fallback solver: runs pytesseract on the entire preprocessed image.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        # you could adjust psm or whitelist here:
        self.tess_config = self.config.get(
            "tesseract_config",
            "--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )

    def solve(self, image_path: str) -> Optional[str]:
        start = time.time()
        try:
            logger.info(f"TesseractSolver: preprocessing {image_path}")
            binary = preprocess_image(image_path)
            # Convert back to PIL-like BGR for pytesseract
            bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            text = pytesseract.image_to_string(bgr, config=self.tess_config).strip()
            elapsed = time.time() - start

            if not text:
                logger.warning("TesseractSolver: no text extracted")
                return None

            logger.info(f"TesseractSolver → '{text}' in {elapsed:.2f}s")
            return text

        except pytesseract.pytesseract.TesseractNotFoundError:
            logger.error("Tesseract binary not found. Please install and configure PATH.")
            return None
        except Exception as e:
            logger.exception(f"TesseractSolver error: {e}")
            return None
