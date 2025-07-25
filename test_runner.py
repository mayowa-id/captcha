"""
CAPTCHA Test Runner - Main test execution
Save as: tests/test_runner.py
"""

import sys
import os
import time
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from captcha_solver.solvers.text_captcha_solver import TextCaptchaSolver

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def fetch_image(path_or_url: str) -> str:
    """Return a local file path for the given image path or URL"""
    if path_or_url.lower().startswith(("http://", "https://")):
        logger.info(f"Downloading CAPTCHA from URL: {path_or_url}")
        if not REQUESTS_AVAILABLE:
            raise Exception("requests library required for URL downloads")
        
        resp = requests.get(path_or_url, timeout=10)
        resp.raise_for_status()
        
        tmp = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=Path(path_or_url).suffix or ".png"
        )
        tmp.write(resp.content)
        tmp.close()
        return tmp.name
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"CAPTCHA file not found: {path_or_url}")
        return path_or_url

def test_with_images(image_paths: List[str], model_class) -> Dict[str, Any]:
    """Test the solver with multiple images and return detailed results"""
    if not image_paths:
        logger.error("No images to test with!")
        return {"results": [], "summary": {"total": 0, "successful": 0, "success_rate": 0}}
    
    # Initialize model and solver
    model = model_class()
    config = {
        "timeout": 30,
        "confidence_threshold": 0.5,  # Lower threshold for testing
        "max_retries": 3
    }
    
    solver = TextCaptchaSolver(model, config)
    
    logger.info(f"🧪 Testing with {len(image_paths)} images...")
    logger.info("=" * 50)
    
    results = []
    temp_files = []  # Track temporary files for cleanup
    
    for i, image_path in enumerate(image_paths, 1):
        logger.info(f"📸 Test {i}/{len(image_paths)}: {os.path.basename(image_path)}")
        logger.info("-" * 30)
        
        local_image_path = image_path
        is_temp = False
        
        try:
            # Handle URLs by downloading them
            if image_path.startswith(('http://', 'https://')):
                local_image_path = fetch_image(image_path)
                temp_files.append(local_image_path)
                is_temp = True
            
            if not os.path.exists(local_image_path):
                logger.error(f"Image not found: {local_image_path}")
                results.append({
                    "image": image_path,
                    "result": None,
                    "success": False,
                    "error": "File not found"
                })
                continue
            
            start_time = time.time()
            result = solver.solve(local_image_path)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if result:
                logger.info(f"SUCCESS: '{result}' (took {processing_time:.2f}s)")
                results.append({
                    "image": image_path,
                    "result": result,
                    "success": True,
                    "time": processing_time
                })
            else:
                logger.info(f" FAILED: No result (took {processing_time:.2f}s)")
                results.append({
                    "image": image_path,
                    "result": None,
                    "success": False,
                    "time": processing_time
                })
                
        except Exception as e:
            logger.error(f"ERROR: {e}")
            results.append({
                "image": image_path,
                "result": None,
                "success": False,
                "error": str(e)
            })
    
    # Cleanup temporary files
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Removed temp file: {temp_file}")
        except Exception as e:
            logger.debug(f"Failed to remove temp file {temp_file}: {e}")
    
    # Calculate summary
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    success_rate = (successful / total * 100) if total > 0 else 0
    
    summary = {
        "total": total,
        "successful": successful,
        "success_rate": success_rate
    }
    
    if successful > 0:
        avg_time = sum(r.get("time", 0) for r in results if r["success"]) / successful
        summary["avg_processing_time"] = avg_time
    
    return {"results": results, "summary": summary}

def print_test_summary(test_results: Dict[str, Any]):
    """Print a formatted summary of test results"""
    results = test_results["results"]
    summary = test_results["summary"]
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    print(f"Success rate: {summary['successful']}/{summary['total']} ({summary['success_rate']:.1f}%)")
    
    if summary['successful'] > 0 and 'avg_processing_time' in summary:
        print(f"Average processing time: {summary['avg_processing_time']:.2f}s")
    
    print("\nDetailed results:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        image_name = os.path.basename(r["image"])
        result_text = r.get("result", r.get("error", "FAILED"))
        print(f"{status} {image_name}: {result_text}")

def test_from_env(model_class) -> Optional[Dict[str, Any]]:
    """Test with image from environment variables"""
    logger.info("🔍 Testing with environment configuration...")
    
    # Load env var
    src = os.getenv("CAPTCHA_IMAGE_PATH") or os.getenv("CAPTCHA_IMAGE_URL")
    if not src:
        logger.warning(" CAPTCHA_IMAGE_PATH or CAPTCHA_IMAGE_URL not set in environment")
        return None
    
    try:
        test_results = test_with_images([src], model_class)
        return test_results
    except Exception as e:
        logger.error(f" Failed to test environment image: {e}")
        return None