"""
Advanced image preprocessing for text CAPTCHA solving:
- Grayscale conversion
- Noise reduction (median blur)
- Adaptive thresholding
- Morphological operations
- Character segmentation helper functions
"""
import cv2
import numpy as np
from typing import List, Tuple

def preprocess_image(img_path: str, enhanced: bool = True) -> np.ndarray:
    """
    Load image from path and apply preprocessing pipeline.
    Returns a binary (0/255) image suitable for OCR or segmentation.
    
    Args:
        img_path: Path to the image file
        enhanced: If True, applies enhanced preprocessing pipeline
    """
    # Load as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    if enhanced:
        return _enhanced_preprocess(img)
    else:
        return _basic_preprocess(img)

def _basic_preprocess(img: np.ndarray) -> np.ndarray:
    """Your original preprocessing pipeline"""
    # 1) Median blur to reduce salt-and-pepper noise
    blurred = cv2.medianBlur(img, 3)

    # 2) Adaptive thresholding for uneven illumination
    thresh = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # 3) Morphological opening to remove small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return opened

def _enhanced_preprocess(img: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing with multiple techniques"""
    
    # 1) Resize if too small (improves OCR accuracy)
    height, width = img.shape
    if height < 40 or width < 100:
        scale = max(40/height, 100/width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # 2) Enhanced noise reduction
    # Start with your median blur
    blurred = cv2.medianBlur(img, 3)
    
    # Add bilateral filter for edge preservation
    bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)

    # 3) Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bilateral)

    # 4) Multiple thresholding approaches - try all and pick best
    # Your original adaptive threshold
    thresh1 = cv2.adaptiveThreshold(
        enhanced,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    
    # OTSU threshold
    _, thresh2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Mean adaptive threshold
    thresh3 = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Choose the best threshold based on character count
    thresholds = [thresh1, thresh2, thresh3]
    best_thresh = _select_best_threshold(thresholds, img.shape)

    # 5) Your morphological operations (enhanced)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(best_thresh, cv2.MORPH_OPEN, kernel)
    
    # Additional cleanup
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2)

    return cleaned

def _select_best_threshold(thresholds: List[np.ndarray], img_shape: Tuple[int, int]) -> np.ndarray:
    """Select the threshold with the most reasonable character count"""
    best_thresh = thresholds[0]  # Default to first
    best_score = 0
    
    for thresh in thresholds:
        # Count potential characters
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_count = 0
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Use reasonable size filters
            if (10 < w < img_shape[1]//2 and 15 < h < img_shape[0] and
                w * h > 100):  # Minimum area
                char_count += 1
        
        # Score based on reasonable character count (3-8 chars typical for CAPTCHA)
        if 3 <= char_count <= 8:
            score = 1.0 / (abs(char_count - 5) + 1)
        elif char_count == 1 or char_count == 2:
            score = 0.7  # Single/double chars can be valid
        else:
            score = 0.1  # Too many or too few
            
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh

def segment_characters(binary_img: np.ndarray, enhanced: bool = True) -> List[np.ndarray]:
    """
    Enhanced character segmentation based on your original code.
    
    Args:
        binary_img: Binary image from preprocessing
        enhanced: If True, uses enhanced segmentation logic
    """
    if enhanced:
        return _enhanced_segment_characters(binary_img)
    else:
        return _basic_segment_characters(binary_img)

def _basic_segment_characters(binary_img: np.ndarray) -> List[np.ndarray]:
    """Your original segmentation function (fixed version)"""
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_regions: List[Tuple[int, int, int, int]] = []  # x, y, w, h

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small noise
        if w < 5 or h < 10:
            continue
        char_regions.append((x, y, w, h))

    # Sort by x-coordinate (left to right)
    char_regions = sorted(char_regions, key=lambda bbox: bbox[0])

    chars: List[np.ndarray] = []
    for x, y, w, h in char_regions:
        char_img = binary_img[y : y + h, x : x + w]
        
        # Your fixed resizing logic
        ratio = 32 / h
        new_w = int(w * ratio)
        
        if new_w > 32:
            new_w = 32
            ratio = 32 / w
            new_h = int(h * ratio)
            if new_h > 32:
                new_h = 32
            resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            resized = cv2.resize(char_img, (new_w, 32), interpolation=cv2.INTER_NEAREST)
        
        # Pad to fixed width (32x32)
        padded = np.zeros((32, 32), dtype=np.uint8)
        
        # Calculate padding for both dimensions
        pad_x = (32 - resized.shape[1]) // 2
        pad_y = (32 - resized.shape[0]) // 2
        
        # Ensure we don't exceed bounds
        end_x = min(pad_x + resized.shape[1], 32)
        end_y = min(pad_y + resized.shape[0], 32)
        
        padded[pad_y:end_y, pad_x:end_x] = resized[:end_y-pad_y, :end_x-pad_x]
        chars.append(padded)

    return chars

def _enhanced_segment_characters(binary_img: np.ndarray) -> List[np.ndarray]:
    """Enhanced segmentation with better filtering and handling"""
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_regions: List[Tuple[int, int, int, int, float]] = []  # x, y, w, h, area

    img_height, img_width = binary_img.shape
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Enhanced filtering based on image dimensions
        min_w = max(5, img_width * 0.02)    # At least 2% of image width
        max_w = min(img_width * 0.4, img_width - 5)  # At most 40% of image width
        min_h = max(10, img_height * 0.3)   # At least 30% of image height
        max_h = img_height * 0.9            # At most 90% of image height
        
        # Filter by aspect ratio (characters shouldn't be too wide or too thin)
        aspect_ratio = w / h if h > 0 else 0
        
        # More sophisticated filtering
        if (min_w <= w <= max_w and min_h <= h <= max_h and 
            0.1 <= aspect_ratio <= 4.0 and area >= 50):
            char_regions.append((x, y, w, h, area))

    # Sort by x-coordinate (left to right)
    char_regions = sorted(char_regions, key=lambda bbox: bbox[0])

    # Filter overlapping regions (sometimes characters get double-detected)
    filtered_regions = []
    for i, (x, y, w, h, area) in enumerate(char_regions):
        is_overlap = False
        for j, (x2, y2, w2, h2, area2) in enumerate(filtered_regions):
            # Check for significant overlap
            overlap_x = max(0, min(x + w, x2 + w2) - max(x, x2))
            overlap_y = max(0, min(y + h, y2 + h2) - max(y, y2))
            overlap_area = overlap_x * overlap_y
            
            if overlap_area > 0.5 * min(w * h, w2 * h2):
                is_overlap = True
                # Keep the larger one
                if area > area2:
                    filtered_regions[j] = (x, y, w, h, area)
                break
        
        if not is_overlap:
            filtered_regions.append((x, y, w, h, area))

    chars: List[np.ndarray] = []
    target_size = 32
    
    for x, y, w, h, _ in filtered_regions:
        char_img = binary_img[y : y + h, x : x + w]
        
        # Enhanced resizing with better interpolation
        scale_w = target_size / w
        scale_h = target_size / h
        scale = min(scale_w, scale_h, 2.0)  # Don't upscale too much
        
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        # Use better interpolation for upscaling
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(char_img, (new_w, new_h), interpolation=interpolation)
        
        # Create padded image (same as your logic)
        padded = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # Center the character
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        
        # Ensure we don't exceed bounds
        end_x = min(pad_x + new_w, target_size)
        end_y = min(pad_y + new_h, target_size)
        
        padded[pad_y:end_y, pad_x:end_x] = resized[:end_y-pad_y, :end_x-pad_x]
        chars.append(padded)

    return chars

# Convenience functions to maintain compatibility
def preprocess_image_basic(img_path: str) -> np.ndarray:
    """Use your original preprocessing"""
    return preprocess_image(img_path, enhanced=False)

def preprocess_image_enhanced(img_path: str) -> np.ndarray:
    """Use enhanced preprocessing"""
    return preprocess_image(img_path, enhanced=True)

def segment_characters_basic(binary_img: np.ndarray) -> List[np.ndarray]:
    """Use your original segmentation"""
    return segment_characters(binary_img, enhanced=False)

def segment_characters_enhanced(binary_img: np.ndarray) -> List[np.ndarray]:
    """Use enhanced segmentation"""
    return segment_characters(binary_img, enhanced=True)