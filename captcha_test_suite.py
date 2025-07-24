import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from captcha_solver.solvers.text_captcha_solver import TextCaptchaSolver
from PIL import Image, ImageDraw, ImageFont
import random
import string
from enhanced_ocr import EnhancedOCR
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR'


try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("requests library not available - install with: pip install requests")

class SimpleCaptchaModel:
    """A simple mock model that uses basic OCR for testing"""
    
    def __init__(self):
        try:
            import pytesseract
            self.ocr_available = True
            print(" OCR (pytesseract) available")
        except ImportError:
            print(" pytesseract not available, using placeholder predictions")
            self.ocr_available = False
    
    def predict(self, image_input):

     from PIL import Image
     import numpy as np
    
    # Use the enhanced OCR class
    ocr = EnhancedOCR()  # Add this class to the file
    text, confidence = ocr.predict_with_confidence(image_input)
    
    # Helper to wrap text & confidence
    class Prediction:
        def __init__(self, text, confidence):
            self.text = text
            self.confidence = confidence
    
            return Prediction(text or "?", confidence)

def create_test_captcha(text="ABC123", filename="test_captcha.png"):
    """Create a simple test CAPTCHA image"""
    try:
        # Create image
        img = Image.new('RGB', (150, 50), 'white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Add some noise/distortion
        for _ in range(20):
            x = random.randint(0, 149)
            y = random.randint(0, 49)
            draw.point((x, y), fill='gray')
        
        # Draw text
        if font:
            draw.text((10, 15), text, fill='black', font=font)
        else:
            draw.text((10, 15), text, fill='black')
        
        # Add some lines for distortion
        for _ in range(3):
            start = (random.randint(0, 149), random.randint(0, 49))
            end = (random.randint(0, 149), random.randint(0, 49))
            draw.line([start, end], fill='gray', width=1)
        
        img.save(filename)
        print(f" Created test CAPTCHA: {filename} (text: {text})")
        return filename
        
    except Exception as e:
        print(f"Error creating test image: {e}")
        return None

def download_sample_images():
    """Download some sample CAPTCHA images from various sources"""
    if not REQUESTS_AVAILABLE:
        print(" requests library needed for downloading. Install with: pip install requests")
        return []
    
    # Sample URLs (these are placeholder/demo images)
    sample_urls = [
        "https://via.placeholder.com/150x50/ffffff/000000?text=HELLO",
        "https://via.placeholder.com/120x40/f0f0f0/333333?text=12345",
        "https://via.placeholder.com/160x60/e0e0e0/000000?text=TEST99"
    ]
    
    downloaded_files = []
    
    for i, url in enumerate(sample_urls):
        try:
            print(f"Downloading sample {i+1}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filename = f"sample_captcha_{i+1}.png"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append(filename)
                print(f"Downloaded: {filename}")
            else:
                print(f"Failed to download sample {i+1}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading sample {i+1}: {e}")
    
    return downloaded_files

def test_with_images(image_paths):
    """Test the solver with multiple images"""
    if not image_paths:
        print("No images to test with!")
        return
    
    # Initialize model and solver
    model = SimpleCaptchaModel()
    config = {
        "timeout": 30,
        "confidence_threshold": 0.5,  # Lower threshold for testing
        "max_retries": 3
    }
    
    solver = TextCaptchaSolver(model, config)
    
    print(f"\nTesting with {len(image_paths)} images...")
    print("=" * 50)
    
    results = []
    
    for i, image_path in enumerate(image_paths, 1):
        if not os.path.exists(image_path):
            print(f"Image {i}: File not found - {image_path}")
            continue
            
        print(f"\n Test {i}: {image_path}")
        print("-" * 30)
        
        try:
            start_time = time.time()
            result = solver.solve(image_path)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if result:
                print(f"SUCCESS: '{result}' (took {processing_time:.2f}s)")
                results.append({"image": image_path, "result": result, "success": True, "time": processing_time})
            else:
                print(f" FAILED: No result (took {processing_time:.2f}s)")
                results.append({"image": image_path, "result": None, "success": False, "time": processing_time})
                
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({"image": image_path, "result": None, "success": False, "error": str(e)})
    
    # Summary
    print("\n" + "=" * 50)
    print(" SUMMARY")
    print("=" * 50)
    
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    if successful > 0:
        avg_time = sum(r["time"] for r in results if r["success"]) / successful
        print(f"Average processing time: {avg_time:.2f}s")
    
    print("\nDetailed results:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"{status} {r['image']}: {r.get('result', 'FAILED')}")

def main():
    """Main function to run CAPTCHA tests"""
    print("CAPTCHA Solver Test Suite")
    print("=" * 50)
    
    test_images = []
    
    # Option 1: Create test images locally
    print("\n1️⃣  Creating local test images...")
    test_texts = ["HELLO", "12345", "ABC123", "TEST99", "XYZ789"]
    for text in test_texts:
        filename = create_test_captcha(text, f"local_{text.lower()}.png")
        if filename:
            test_images.append(filename)
    
    # Option 2: Download sample images
    print("\n2️⃣  Downloading sample images...")
    downloaded = download_sample_images()
    test_images.extend(downloaded)
    
    # Option 3: Check for user-provided images
    print("\n3️⃣  Checking for user images...")
    user_images = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) 
                   and 'captcha' in f.lower() and f not in test_images]
    if user_images:
        print(f"Found user images: {user_images}")
        test_images.extend(user_images)
    
    # Run tests
    if test_images:
        test_with_images(test_images)
    else:
        print(" No images available for testing!")
        print("\nTo test with your own images:")
        print("1. Put CAPTCHA images in this directory")
        print("2. Make sure filenames contain 'captcha'")
        print("3. Run this script again")
    
    # Cleanup
    print(f"\n Cleaning up temporary files...")
    cleanup_files = [f for f in test_images if f.startswith(('local_', 'sample_captcha_'))]
    for f in cleanup_files:
        try:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed: {f}")
        except:
            pass

if __name__ == "__main__":
    main()