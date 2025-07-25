import os
import sys
import tempfile
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from captcha_solver.solvers.text_captcha_solver import TextCaptchaSolver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()  # loads .env into os.environ

def fetch_image(path_or_url: str) -> str:
    """Return a local file path for the given image path or URL."""
    if path_or_url.lower().startswith(("http://", "https://")):
        logger.info(f"Downloading CAPTCHA from URL: {path_or_url}")
        resp = requests.get(path_or_url, timeout=10)
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(path_or_url).suffix or ".png")
        tmp.write(resp.content)
        tmp.close()
        return tmp.name
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"CAPTCHA file not found: {path_or_url}")
        return path_or_url

class SimpleCaptchaModel:
    """Mock model using pytesseract if available, else placeholder."""
    class Prediction:
        def __init__(self, text: str, confidence: float):
            self.text = text
            self.confidence = confidence

    def __init__(self):
        try:
            import pytesseract  # noqa: F401
            self.ocr = True
        except ImportError:
            logger.warning("pytesseract not found: falling back to placeholder")
            self.ocr = False

    def predict(self, image_path: str) -> Prediction:
        if self.ocr:
            from pytesseract import image_to_string
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            txt = image_to_string(
                img,
                config="--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            ).strip()
            conf = 0.8 if txt.isalnum() and len(txt) > 2 else 0.15
            return SimpleCaptchaModel.Prediction(txt, conf)
        else:
            return SimpleCaptchaModel.Prediction("TEST123", 0.85)

def test_real_captcha():
    # Load env var
    src = os.getenv("CAPTCHA_IMAGE_PATH") or os.getenv("CAPTCHA_IMAGE_URL")
    if not src:
        logger.error("Define CAPTCHA_IMAGE_PATH or CAPTCHA_IMAGE_URL in your .env")
        sys.exit(1)

    # Fetch (or validate) the image
    try:
        image_file = fetch_image(src)
    except Exception as e:
        logger.error(f"Failed to obtain CAPTCHA image: {e}")
        sys.exit(1)

    try:
        # Initialize and run solver
        model = SimpleCaptchaModel()
        cfg = {"timeout": 30, "confidence_threshold": 0.7, "max_retries": 3}
        solver = TextCaptchaSolver(model, cfg)

        logger.info(f"Solving CAPTCHA at: {image_file}")
        answer = solver.solve(image_file)

        if answer:
            logger.info(f"Solved CAPTCHA: {answer}")
        else:
            logger.error("CAPTCHA solving failed (low confidence or error)")

    finally:
        # Cleanup temp if we downloaded
        if image_file != src and os.path.exists(image_file):
            os.remove(image_file)
            logger.debug(f"Removed temp file: {image_file}")

if __name__ == "__main__":
    test_real_captcha()
