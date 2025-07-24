"""
Main CAPTCHA Test Suite
Save as: main_captcha_test.py (in your project root)
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print(" python-dotenv not available - install with: pip install python-dotenv")

# Import our modular components
from tests.test_runner import test_from_env, test_with_images, print_test_summary

# Try to import enhanced model
try:
    from captcha_solver.models.enhanced_captcha_model import EnhancedCaptchaModel
    MODEL_CLASS = EnhancedCaptchaModel
except ImportError:
    # Fallback to your existing model structure
    try:
        from tests.test_real_captcha import SimpleCaptchaModel
        MODEL_CLASS = SimpleCaptchaModel
    except ImportError:
        print(" No captcha model found. Please ensure your model is properly set up.")
        sys.exit(1)

# Try to import Kaggle integration
try:
    from captcha_solver.integrations.kaggle_integration import KaggleIntegration
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# Try to import utilities
try:
    from captcha_solver.utils.image_generators import create_test_captcha, download_sample_images
except ImportError:
    # Fallback implementations
    def create_test_captcha(text="ABC123", filename="test_captcha.png"):
        """Fallback test image creation"""
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (150, 50), 'white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 15), text, fill='black')
            img.save(filename)
            print(f"Created test CAPTCHA: {filename}")
            return filename
        except Exception as e:
            print(f"Error creating test image: {e}")
            return None
    
    def download_sample_images():
        """Fallback sample download"""
        print("  Sample image download not available")
        return []

logger = logging.getLogger(__name__)

def main():
    """Main function to run comprehensive CAPTCHA tests"""
    print(" CAPTCHA Solver Test Suite")
    print("=" * 50)
    
    all_test_images = []
    
    # Test 1: Environment variable (if available)
    print("\n1️⃣  Testing with environment configuration...")
    env_results = test_from_env(MODEL_CLASS)
    if env_results:
        print_test_summary(env_results)
    
    # Test 2: Kaggle dataset integration
    if KAGGLE_AVAILABLE:
        print("\n2️⃣  Testing with Kaggle CAPTCHA dataset...")
        try:
            kaggle_integration = KaggleIntegration()
            if kaggle_integration.is_available():
                kaggle_images = kaggle_integration.get_random_captcha_images(count=5)
                if kaggle_images:
                    all_test_images.extend(kaggle_images)
                    logger.info(f"✅ Added {len(kaggle_images)} Kaggle images")
                else:
                    logger.warning("⚠️  No Kaggle images available")
            else:
                print("⚠️  Kaggle credentials not configured")
        except Exception as e:
            print(f" Kaggle integration failed: {e}")
    else:
        print("\n Kaggle integration not available")
        print("   To enable: pip install kaggle")
        print("   Set KAGGLE_USERNAME and KAGGLE_KEY in .env")
    
    # Test 3: Create local test images
    print("\n3️⃣  Creating local test images...")
    test_texts = ["HELLO", "12345", "ABC123", "TEST99", "XYZ789"]
    local_images = []
    for text in test_texts:
        filename = create_test_captcha(text, f"local_{text.lower()}.png")
        if filename:
            local_images.append(filename)
    
    all_test_images.extend(local_images)
    
    # Test 4: Download sample images
    print("\n4  Downloading sample images...")
    try:
        downloaded = download_sample_images()
        all_test_images.extend(downloaded)
    except Exception as e:
        print(f"⚠️  Sample download failed: {e}")
    
    # Test 5: Check for user-provided images
    print("\n5 Checking for user images...")
    user_images = [
        f for f in os.listdir('.')
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        and 'captcha' in f.lower()
        and f not in [os.path.basename(img) for img in all_test_images]
    ]
    if user_images:
        print(f"Found user images: {user_images}")
        all_test_images.extend(user_images)
    
    # Run comprehensive tests
    if all_test_images:
        print(f"\n🧪 Running comprehensive test with {len(all_test_images)} images...")
        final_results = test_with_images(all_test_images, MODEL_CLASS)
        print_test_summary(final_results)
    else:
        print(" No images available for testing!")
        print("\n💡 To enable more tests:")
        print("   • Set CAPTCHA_IMAGE_PATH or CAPTCHA_IMAGE_URL in .env")
        print("   • Set up Kaggle credentials (KAGGLE_USERNAME, KAGGLE_KEY)")
        print("   • Put CAPTCHA images in this directory")
        print("   • Install missing dependencies: pip install kaggle requests pillow pytesseract python-dotenv")
    
    # Cleanup temporary files
    print(f"\n🧹 Cleaning up temporary files...")
    cleanup_files = [
        f for f in all_test_images
        if os.path.basename(f).startswith(('local_', 'sample_captcha_'))
        and os.path.exists(f)
    ]
    for f in cleanup_files:
        try:
            os.remove(f)
            print(f"Removed: {os.path.basename(f)}")
        except Exception as e:
            logger.debug(f"Failed to remove {f}: {e}")

if __name__ == "__main__":
    main()