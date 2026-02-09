"""
Final automated creative analysis pipeline
Correct approach: OCR + GPT-4.1 visual + Saliency + GPT-5.2 recommendations

Usage: python3 analyze_creative_final.py <image_path>
"""

import sys
import os
import json
import base64
import io
import requests
import torch
import numpy as np
import cv2
import easyocr
from PIL import Image, ImageDraw, ImageFont
from deepgaze_pytorch import DeepGazeIIE
from scipy.special import logsumexp

# Load API key (prioritize environment variables for cloud deployment)
API_KEY = os.environ.get('OPENAI_API_KEY', '')

# If not in environment, try importing from config.py (local development)
if not API_KEY:
    try:
        from config import OPENAI_API_KEY as API_KEY
    except ImportError:
        print("⚠️  Warning: API key not found. For Streamlit Cloud, add OPENAI_API_KEY to Secrets. For local, copy config.example.py to config.py and add your API key.")
        API_KEY = ''

# Maximum image dimension to prevent OOM errors
# Reduced for Streamlit Cloud (1GB RAM limit: DeepGaze ~500MB + PyTorch ~200MB)
MAX_IMAGE_DIMENSION = 600

# Global model caches — avoid reloading on each analysis
_deepgaze_model = None
_easyocr_reader = None


def get_deepgaze_model():
    """Get cached DeepGaze model (lazy loading, ~500MB)"""
    global _deepgaze_model
    if _deepgaze_model is None:
        print("  Loading DeepGaze model (first time, will be cached)...")
        _deepgaze_model = DeepGazeIIE(pretrained=True)
        _deepgaze_model.eval()
    return _deepgaze_model


def get_easyocr_reader():
    """Get cached EasyOCR reader (lazy loading)"""
    global _easyocr_reader
    if _easyocr_reader is None:
        print("  Loading EasyOCR model (first time, will be cached)...")
        _easyocr_reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
    return _easyocr_reader


def resize_image_if_needed(img, max_dim=MAX_IMAGE_DIMENSION):
    """Resize image if larger than max_dim to prevent OOM errors.
    
    Returns:
        tuple: (resized_img, scale_factor) where scale_factor is used to 
               scale coordinates back to original size
    """
    width, height = img.size
    if max(width, height) <= max_dim:
        return img, 1.0
    
    scale = max_dim / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    resized = img.resize(new_size, Image.LANCZOS)
    print(f"  ⚠️ Resized image: {width}x{height} → {new_size[0]}x{new_size[1]}")
    return resized, scale


def print_step(step, title):
    print(f"\n{'='*70}")
    print(f"STEP {step}: {title}")
    print('='*70)

# ============================================================================
# STEP 1: Generate Saliency Map
# ============================================================================

def generate_saliency_map(image_path):
    """Generate saliency map using DeepGaze"""
    print("  Generating saliency map with DeepGaze...")

    img_original = Image.open(image_path).convert('RGB')
    original_width, original_height = img_original.size
    
    # Resize if needed to prevent OOM
    img_resized, scale = resize_image_if_needed(img_original)
    img_array_resized = np.array(img_resized)
    height, width = img_array_resized.shape[:2]
    
    # Free resized PIL image (keep only numpy array)
    del img_resized

    # Use cached model to avoid reloading (~500MB)
    model = get_deepgaze_model()
    image_tensor = torch.from_numpy(img_array_resized.transpose(2, 0, 1)[None, ...]).float()

    centerbias = np.zeros((height, width))
    centerbias -= logsumexp(centerbias)
    centerbias_tensor = torch.from_numpy(centerbias[None, ...]).float()

    with torch.no_grad():
        log_density = model(image_tensor, centerbias_tensor)
    
    # Free tensors immediately
    del image_tensor, centerbias_tensor
    
    saliency_map = log_density.exp().numpy()[0, 0]
    del log_density
    
    # Resize saliency map back to original dimensions if we resized
    if scale < 1.0:
        saliency_map = cv2.resize(saliency_map, (original_width, original_height), 
                                   interpolation=cv2.INTER_LINEAR)
        # Use original image for output
        img_array = np.array(img_original)
        print(f"  ✅ Saliency map generated at {width}x{height}, scaled to {original_width}x{original_height}")
    else:
        img_array = img_array_resized
        print(f"  ✅ Saliency map generated ({width}x{height})")
    
    del img_original
    return img_array, saliency_map

# ============================================================================
# STEP 2: Detect Text Elements with OCR
# ============================================================================

def detect_text_blocks(image_path):
    """Detect text blocks using EasyOCR with preprocessing"""
    print("  Detecting text blocks with EasyOCR...")

    # Use cached reader to avoid reloading models
    reader = get_easyocr_reader()
    img_original = Image.open(image_path)
    
    # Resize if needed to prevent OOM
    img_resized, scale = resize_image_if_needed(img_original)
    img_array = np.array(img_resized)
    
    # Preprocess image for better OCR on bright/colored backgrounds
    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Try OCR on both original and enhanced images
    results_original = reader.readtext(img_array)
    results_enhanced = reader.readtext(enhanced)
    
    # Merge results, preferring higher confidence
    all_results = {}
    
    for bbox, text, conf in results_original + results_enhanced:
        text_clean = text.strip()
        if text_clean and len(text_clean) >= 2:
            # Use text as key, keep highest confidence
            if text_clean not in all_results or conf > all_results[text_clean][1]:
                all_results[text_clean] = (bbox, conf)
    
    text_blocks = []
    for text, (bbox, conf) in all_results.items():
        # Get bounding box coordinates
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]

        x = int(min(x_coords))
        y = int(min(y_coords))
        w = int(max(x_coords) - x)
        h = int(max(y_coords) - y)
        
        # Scale coordinates back to original image size
        if scale < 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

        # Lower threshold for more coverage
        if conf < 0.2 or len(text) < 2:
            continue

        text_blocks.append({
            'text': text,
            'bbox': [x, y, w, h],
            'confidence': float(conf)
        })
        print(f"    OCR: '{text}' (conf: {conf:.2f})")

    print(f"  ✅ Found {len(text_blocks)} text blocks")
    return text_blocks

# ============================================================================
# STEP 3: Group and Classify Text Zones
# ============================================================================

def group_and_classify_text_zones(text_blocks, img_width, img_height):
    """Group text blocks into semantic zones"""
    print("  Grouping and classifying text zones...")

    zones = []

    # Separate blocks by position and size
    header_blocks = []
    cta_candidates = []
    subheader_candidates = []
    description_candidates = []
    legal_candidates = []

    for block in text_blocks:
        text = block['text'].lower()
        x, y, w, h = block['bbox']

        # Legal text (bottom, small)
        if y > img_height - 100 and h < 20:
            legal_candidates.append(block)
            continue

        # Logo text (top-left, small)
        if y < 80 and x < 200 and ('пэй' in text or 'pay' in text or 'маркет' in text or 'market' in text):
            # Will be handled as visual element
            continue

        # CTA keywords
        if any(kw in text for kw in ['раскрутите', 'раскрутить', 'купить', 'выбрать', 'получить', 'смотреть', 'перейти']):
            cta_candidates.append(block)
            continue

        # Header (large text, upper-left area)
        if x < img_width * 0.6 and y < img_height * 0.6 and h > 35:
            header_blocks.append(block)
            continue

        # Subheader (right side, medium)
        if x > img_width * 0.6 and 20 < h < 40:
            subheader_candidates.append(block)
            continue

        # Description (right side, smaller)
        if x > img_width * 0.6 and 15 < h < 25:
            description_candidates.append(block)
            continue

    # Merge header blocks (close together)
    if header_blocks:
        header_blocks.sort(key=lambda b: b['bbox'][1])  # Sort by Y

        # Merge blocks that are close vertically
        merged_header = []
        current_group = [header_blocks[0]]

        for i in range(1, len(header_blocks)):
            prev_y = current_group[-1]['bbox'][1]
            prev_h = current_group[-1]['bbox'][3]
            curr_y = header_blocks[i]['bbox'][1]

            # If blocks are close (within 60px)
            if curr_y - (prev_y + prev_h) < 60:
                current_group.append(header_blocks[i])
            else:
                # Save current group
                if current_group:
                    merged_header.append(current_group)
                current_group = [header_blocks[i]]

        if current_group:
            merged_header.append(current_group)

        # Take the largest group as header
        if merged_header:
            main_header_group = max(merged_header, key=len)

            min_x = min([b['bbox'][0] for b in main_header_group])
            min_y = min([b['bbox'][1] for b in main_header_group])
            max_x = max([b['bbox'][0] + b['bbox'][2] for b in main_header_group])
            max_y = max([b['bbox'][1] + b['bbox'][3] for b in main_header_group])

            header_text = ' '.join([b['text'] for b in main_header_group])

            zones.append({
                'type': 'header',
                'label': header_text,
                'bbox': [min_x, min_y, max_x - min_x, max_y - min_y]
            })

    # CTA
    if cta_candidates:
        # Take the one with highest confidence
        cta = max(cta_candidates, key=lambda b: b['confidence'])
        zones.append({
            'type': 'cta',
            'label': cta['text'],
            'bbox': cta['bbox']
        })

    # Subheader (merge if multiple)
    if subheader_candidates:
        if len(subheader_candidates) > 1:
            min_x = min([b['bbox'][0] for b in subheader_candidates])
            min_y = min([b['bbox'][1] for b in subheader_candidates])
            max_x = max([b['bbox'][0] + b['bbox'][2] for b in subheader_candidates])
            max_y = max([b['bbox'][1] + b['bbox'][3] for b in subheader_candidates])

            subheader_text = ' '.join([b['text'] for b in subheader_candidates])

            zones.append({
                'type': 'subheader',
                'label': subheader_text,
                'bbox': [min_x, min_y, max_x - min_x, max_y - min_y]
            })
        else:
            zones.append({
                'type': 'subheader',
                'label': subheader_candidates[0]['text'],
                'bbox': subheader_candidates[0]['bbox']
            })

    # Description
    if description_candidates:
        desc = description_candidates[0]
        zones.append({
            'type': 'description',
            'label': desc['text'],
            'bbox': desc['bbox']
        })

    # Legal (merge all legal text)
    if legal_candidates and len(legal_candidates) > 0:
        min_x = min([b['bbox'][0] for b in legal_candidates])
        min_y = min([b['bbox'][1] for b in legal_candidates])
        max_x = max([b['bbox'][0] + b['bbox'][2] for b in legal_candidates])
        max_y = max([b['bbox'][1] + b['bbox'][3] for b in legal_candidates])

        zones.append({
            'type': 'legal',
            'label': 'Юридическая информация',
            'bbox': [min_x, min_y, max_x - min_x, max_y - min_y]
        })

    print(f"  ✅ Classified {len(zones)} text zones")
    return zones

# ============================================================================
# STEP 4: Detect Visual Elements with GPT-4.1
# ============================================================================

def detect_visual_elements_gpt41(image_path, existing_zones, img_width, img_height):
    """Detect visual elements and missing text zones using GPT-5.2 with reasoning"""
    print("  Detecting elements with GPT-5.2 + reasoning...")

    with open(image_path, 'rb') as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    image_format = "png" if image_path.endswith('.png') else "jpeg"

    # Describe existing zones to avoid duplication
    existing_desc = "\n".join([f"- {z['type']}: {z['bbox']}" for z in existing_zones])

    # Check if we have text zones from OCR
    has_text_zones = any(z['type'] in ['header', 'subheader', 'cta', 'slogan', 'description'] for z in existing_zones)
    
    if has_text_zones:
        # OCR found text - GPT only needs to find visual elements
        prompt = f"""Найди ТОЛЬКО ВИЗУАЛЬНЫЕ элементы на рекламном креативе.

Размер изображения: {img_width}x{img_height} пикселей

Текстовые элементы УЖЕ НАЙДЕНЫ, не детектируй их:
{existing_desc}

**Найди ТОЛЬКО:**
- "logo" — логотип бренда (маленький, обычно в углу)
- "person" — человек/лицо на изображении (обведи ТОЛЬКО лицо или фигуру, НЕ захватывай текст!)
- "product" — изображение продукта/товара (НЕ человек)

**КРИТИЧЕСКИ ВАЖНО для person:**
- Обводи ТОЛЬКО человека/лицо
- НЕ включай текст в bbox человека
- Если человек частично за текстом — обведи только видимую часть человека

**Формат bbox:** [x, y, width, height] в пикселях

Верни ТОЛЬКО JSON массив:
[
  {{"type": "person", "label": "описание", "bbox": [x, y, width, height]}},
  {{"type": "logo", "label": "название", "bbox": [x, y, width, height]}}
]"""
    else:
        # No OCR text - GPT needs to find everything
        prompt = f"""Найди ВСЕ ключевые элементы на рекламном креативе.

Размер изображения: {img_width}x{img_height} пикселей

**Типы элементов:**

ВИЗУАЛЬНЫЕ:
- "logo" — логотип бренда (маленький, в углу)
- "person" — человек/лицо (обведи ТОЛЬКО человека, не текст вокруг!)
- "product" — изображение продукта (НЕ человек)

ТЕКСТОВЫЕ:
- "header" — главный заголовок/оффер (самый крупный текст)
- "subheader" — подзаголовок
- "cta" — кнопка призыва к действию (прямоугольник с текстом)
- "slogan" — слоган, дополнительный текст

**Формат bbox:** [x, y, width, height] в пикселях
- x, y — левый верхний угол
- bbox должен ТОЧНО обрамлять элемент

Верни ТОЛЬКО JSON массив:
[
  {{"type": "header", "label": "текст", "bbox": [x, y, width, height]}},
  {{"type": "person", "label": "описание", "bbox": [x, y, width, height]}},
  ...
]"""

    payload = {
        'model': 'gpt-5.2',
        'messages': [
            {
                'role': 'system',
                'content': 'Ты эксперт по анализу рекламных креативов. Твоя задача — точно определить координаты элементов на изображении. Будь внимателен к деталям и давай ТОЧНЫЕ пиксельные координаты.'
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/{image_format};base64,{base64_image}'
                        }
                    }
                ]
            }
        ],
        'max_completion_tokens': 2000,
        'reasoning_effort': 'medium'
    }

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        },
        json=payload,
        timeout=120  # Longer timeout for reasoning
    )

    if response.status_code != 200:
        print(f"  ⚠️  GPT-5.2 error: {response.status_code} - {response.text}")
        return []

    result = response.json()
    text = result['choices'][0]['message']['content'].strip()

    # Parse JSON
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1])
        if text.startswith('json'):
            text = text[4:].strip()

    try:
        visual_zones = json.loads(text)
        
        # Post-process: remove duplicates and validate
        visual_zones = postprocess_zones(visual_zones, img_width, img_height)
        
        print(f"  ✅ Found {len(visual_zones)} elements after validation")
        return visual_zones
    except:
        print(f"  ⚠️  Failed to parse GPT-4.1 response")
        return []


def postprocess_zones(zones, img_width, img_height):
    """Validate and clean up detected zones"""
    
    def bbox_overlap_pct(bbox1, bbox2):
        """Calculate overlap percentage between two bboxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        
        return intersection / min(area1, area2) if min(area1, area2) > 0 else 0
    
    validated = []
    
    for zone in zones:
        # Skip if no bbox
        if 'bbox' not in zone or len(zone['bbox']) != 4:
            continue
            
        x, y, w, h = zone['bbox']
        
        # Validate bbox is within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(10, min(w, img_width - x))
        h = max(10, min(h, img_height - y))
        
        zone['bbox'] = [int(x), int(y), int(w), int(h)]
        
        # Check for duplicates (same area, different type)
        is_duplicate = False
        for existing in validated:
            overlap = bbox_overlap_pct(zone['bbox'], existing['bbox'])
            if overlap > 0.7:
                # If logo and header overlap, keep only logo
                if zone['type'] == 'header' and existing['type'] == 'logo':
                    is_duplicate = True
                    break
                elif zone['type'] == 'logo' and existing['type'] == 'header':
                    # Remove existing header, add logo
                    validated.remove(existing)
                    break
                # If same type overlaps, skip duplicate
                elif zone['type'] == existing['type']:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            validated.append(zone)
    
    return validated

# ============================================================================
# STEP 5: Refine CTA bbox (find button)
# ============================================================================

def refine_cta_bbox(image_path, cta_zone):
    """Refine CTA bbox by finding the button around the text"""
    print("  Refining CTA bbox...")

    if not cta_zone:
        return cta_zone

    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x, y, w, h = cta_zone['bbox']

    # Expand search area around text
    padding = 30
    search_x1 = max(0, x - padding)
    search_y1 = max(0, y - padding)
    search_x2 = min(img.shape[1], x + w + padding)
    search_y2 = min(img.shape[0], y + h + padding)

    # Try to find dark button region
    roi = gray[search_y1:search_y2, search_x1:search_x2]

    # Threshold to find dark regions
    _, thresh = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest_contour)

        # Convert back to image coordinates
        button_x = search_x1 + bx
        button_y = search_y1 + by

        # Check if this is reasonable (not too far from text)
        if abs(button_x - x) < 50 and abs(button_y - y) < 50 and bw > w * 0.8:
            cta_zone['bbox'] = [button_x, button_y, bw, bh]
            print(f"  ✅ Refined CTA bbox to button area")
            return cta_zone

    # Fallback: expand text bbox by 20px on each side
    expanded_x = max(0, x - 20)
    expanded_y = max(0, y - 15)
    expanded_w = w + 40
    expanded_h = h + 30

    cta_zone['bbox'] = [expanded_x, expanded_y, expanded_w, expanded_h]
    print(f"  ✅ Expanded CTA bbox (fallback)")
    return cta_zone

# ============================================================================
# STEP 6: Merge All Zones
# ============================================================================

def merge_all_zones(text_zones, visual_zones, filter_legal=False):
    """Merge text and visual zones, remove duplicates"""
    print("  Merging all zones...")

    all_zones = text_zones + visual_zones

    # Filter legal if needed
    if filter_legal:
        all_zones = [z for z in all_zones if z['type'] != 'legal']

    # Remove duplicates (overlap > 80%)
    def bbox_overlap(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2

        return intersection / min(area1, area2)

    def is_nested(zone1, zone2):
        """Check if zone2 is nested inside zone1"""
        x1, y1, w1, h1 = zone1['bbox']
        x2, y2, w2, h2 = zone2['bbox']

        # zone2 is nested if it's completely inside zone1
        return (x2 >= x1 and y2 >= y1 and
                x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1)

    # Remove duplicates, but keep nested zones (e.g., subheader inside product)
    unique_zones = []
    for zone in all_zones:
        is_duplicate = False
        for existing in unique_zones:
            overlap = bbox_overlap(zone['bbox'], existing['bbox'])

            # If overlap > 80%, check if it's a nested relationship
            if overlap > 0.8:
                # If one is nested in the other, keep both
                if is_nested(zone, existing) or is_nested(existing, zone):
                    continue  # Not a duplicate, it's parent-child
                else:
                    # True duplicate - same type or truly overlapping
                    if zone['type'] == existing['type']:
                        is_duplicate = True
                        break

        if not is_duplicate:
            unique_zones.append(zone)

    print(f"  ✅ Merged to {len(unique_zones)} unique zones")
    return unique_zones

# ============================================================================
# STEP 7: Calculate Attention per Zone
# ============================================================================

def calculate_attention(saliency_map, zones):
    """Calculate attention percentage for each zone"""
    print("  Calculating attention for zones...")

    height, width = saliency_map.shape
    total_saliency = saliency_map.sum()

    def is_nested_in(zone1, zone2):
        """Check if zone1 is nested inside zone2"""
        x1, y1, w1, h1 = zone1['bbox']
        x2, y2, w2, h2 = zone2['bbox']

        return (x1 >= x2 and y1 >= y2 and
                x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2)

    zones_with_attention = []

    for zone in zones:
        x, y, w, h = zone['bbox']

        # Clip to image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        # Get zone saliency
        zone_mask = np.zeros_like(saliency_map, dtype=bool)
        zone_mask[y:y+h, x:x+w] = True

        # Find nested zones (children)
        nested_zones = []
        for other_zone in zones:
            if other_zone != zone and is_nested_in(other_zone, zone):
                nested_zones.append(other_zone)

        # Subtract nested zones from parent
        if nested_zones:
            for nested in nested_zones:
                nx, ny, nw, nh = nested['bbox']
                nx = max(0, min(nx, width - 1))
                ny = max(0, min(ny, height - 1))
                nw = max(1, min(nw, width - nx))
                nh = max(1, min(nh, height - ny))

                # Remove nested area from parent mask
                zone_mask[ny:ny+nh, nx:nx+nw] = False

        # Calculate attention only for non-nested area
        zone_attention = saliency_map[zone_mask].sum()
        attention_pct = (zone_attention / total_saliency) * 100

        zones_with_attention.append({
            **zone,
            'attention_pct': float(round(attention_pct, 1))
        })

    # Sort by attention (highest first)
    zones_with_attention.sort(key=lambda z: z['attention_pct'], reverse=True)

    total_zones_attention = sum([z['attention_pct'] for z in zones_with_attention])

    print(f"  ✅ Calculated attention for {len(zones_with_attention)} zones")
    print(f"  Total coverage: {total_zones_attention:.1f}%")

    return zones_with_attention, total_zones_attention

# ============================================================================
# STEP 8: Generate Recommendations
# ============================================================================

def generate_recommendations(zones, total_zones_attention, background_attention):
    """Generate recommendations using GPT-5.2"""
    print("  Generating recommendations with GPT-5.2...")

    zones_summary = []
    for zone in zones:
        zones_summary.append({
            'type': zone['type'],
            'label': zone['label'][:50],  # Truncate long labels
            'attention_pct': zone['attention_pct']
        })

    prompt = f"""Проанализируй результаты eye-tracking анализа медийного баннера.

**Данные анализа зон:**
{json.dumps(zones_summary, indent=2, ensure_ascii=False)}

**Покрытие:**
- Контентные зоны: {total_zones_attention}%
- Фон/пустое пространство: {background_attention}%

---

## КРИТЕРИИ ОЦЕНКИ БАННЕРА
(на основе профессиональных стандартов Яндекс Рекламы)

### 1. ЧЁТКОЕ УТП (Уникальное Торговое Предложение)
- Одно главное сообщение, а не несколько
- Минимум информационного шума
- Понятно, что предлагается и почему это ценно
- Header должен получать значительное внимание

### 2. ВИЗУАЛЬНАЯ ИЕРАРХИЯ
- Элементы расположены по убыванию важности
- Путь взгляда: от главного к второстепенному
- Логотип заметен, но не конкурирует с оффером
- Продукт/услуга в центре внимания

### 3. ПРИЗЫВ К ДЕЙСТВИЮ (CTA)
- Чёткая формулировка: "Закажите", "Попробуйте", "Узнайте"
- Яркий и заметный элемент
- Рекомендуемый размер: ~10% от площади баннера
- CTA должен быть частью основного пути взгляда

### 4. НАГЛЯДНОСТЬ
- Показано, как продукт решает задачу
- Акцент на преимуществах и возможностях
- Изображения и текст работают вместе

### 5. БАЛАНС И "ВОЗДУХ"
- Белое пространство создаёт баланс
- Креатив не перегружен элементами
- Важные элементы выделены за счёт пространства вокруг

### 6. ЧЕЛОВЕК/ПЕРСОНА (если есть)
- Лицо привлекает внимание — это нормально
- Важно: направляет ли персона внимание на оффер или отвлекает
- Взгляд персоны может "вести" к CTA или продукту

---

## ОЦЕНКА

На основе данных eye-tracking и критериев выше, оцени:

1. **Overall Score (1-5)**:
   - 5 = Отлично: чёткое УТП, правильная иерархия, заметный CTA, баланс
   - 4 = Хорошо: небольшие улучшения
   - 3 = Средне: есть проблемы с иерархией или CTA
   - 2 = Плохо: нечёткое сообщение, плохая иерархия
   - 1 = Критично: внимание не на ключевых элементах

2. **Reasoning** — 2-3 предложения, объясняющих оценку

3. **Рекомендации (3-5 штук)** — конкретные улучшения:
   - priority: "High" / "Medium" / "Low"
   - title: короткий заголовок
   - description: что и как улучшить
   - expected_impact: ожидаемый эффект

Верни ТОЛЬКО JSON:
{{
  "overall_score": 3.5,
  "reasoning": "объяснение",
  "recommendations": [
    {{
      "priority": "High",
      "title": "заголовок",
      "description": "описание",
      "expected_impact": "эффект"
    }}
  ]
}}"""

    payload = {
        'model': 'gpt-5.2',
        'messages': [
            {
                'role': 'system',
                'content': 'Ты эксперт по медийной рекламе и дизайну баннеров с опытом работы в ведущих рекламных агентствах (REDKEDS, ИКРА, FABULA). Ты знаешь профессиональные стандарты создания эффективных баннеров и даёшь практичные рекомендации на основе данных eye-tracking.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'max_completion_tokens': 2000,
        'reasoning_effort': 'medium'
    }

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        },
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"GPT-5.2 API error: {response.status_code} - {response.text}")

    result = response.json()
    text = result['choices'][0]['message']['content'].strip()

    # Parse JSON
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1])
        if text.startswith('json'):
            text = text[4:].strip()

    recommendations = json.loads(text)

    print(f"  ✅ Generated {len(recommendations['recommendations'])} recommendations")
    print(f"  Overall Score: {recommendations['overall_score']}/5.0")

    return recommendations

# ============================================================================
# STEP 10: Build Edit Prompt for Regeneration
# ============================================================================

def build_edit_prompt(zones, recommendations, img_width, img_height):
    """Build a structured edit prompt for GPT Image using GPT-5.2"""
    print("  Building edit prompt with GPT-5.2...")

    # Filter: only High and Medium priority recommendations (case-insensitive)
    for r in recommendations:
        print(f"    Rec priority='{r.get('priority')}' title='{r.get('title', '')[:40]}'")
    filtered_recs = [
        r for r in recommendations
        if str(r.get('priority', '')).strip().lower() in ('high', 'medium')
    ]

    if not filtered_recs:
        print("  ⚠️ No High/Medium recommendations — skipping regeneration")
        return None

    print(f"  Found {len(filtered_recs)} High/Medium recommendations")

    zones_summary = []
    for zone in zones:
        zones_summary.append({
            'type': zone['type'],
            'label': zone['label'][:50],
            'bbox': zone['bbox'],
            'attention_pct': zone['attention_pct']
        })

    recs_text = "\n".join([
        f"- [{r['priority']}] {r['title']}: {r['description']}"
        for r in filtered_recs
    ])

    prompt = f"""Ты — промпт-инженер. Твоя задача — превратить рекомендации по улучшению рекламного баннера в ТОЧНУЮ инструкцию для AI-модели редактирования изображений (GPT Image edit).

**Размер изображения:** {img_width}x{img_height} пикселей

**Текущие зоны на баннере:**
{json.dumps(zones_summary, indent=2, ensure_ascii=False)}

**Рекомендации по улучшению:**
{recs_text}

---

## ТВОЯ ЗАДАЧА

Сформируй JSON с инструкцией для редактирования:

1. **edit_prompt** — промпт на АНГЛИЙСКОМ для GPT Image edit. Чёткая инструкция что изменить визуально.

2. **preserve** — список того, что НЕЛЬЗЯ менять.

## КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА

1. **СОХРАНИТЬ ВЕСЬ ТЕКСТ КАК ЕСТЬ.** Все надписи на баннере должны остаться ТОЧНО такими же — тот же язык, те же слова, тот же шрифт. НЕ переводить, НЕ менять, НЕ удалять текст.
2. **СОХРАНИТЬ ЯЗЫК.** Если баннер на русском — текст ОБЯЗАН остаться на русском. Если на английском — на английском.
3. **СОХРАНИТЬ ЛОГОТИП** точно как есть — позиция, стиль, написание.
4. **СОХРАНИТЬ ЛЮДЕЙ/ЛИЦА** если есть — не менять внешность, позу, одежду.
5. **НЕ ДОБАВЛЯТЬ НОВЫЕ ЭЛЕМЕНТЫ** — никаких сертификатов, бейджей, иконок, дополнительных надписей.
6. **ТОЛЬКО ВИЗУАЛЬНЫЕ ПРАВКИ** — менять можно: размер/контраст/яркость элементов, расположение, фон, цветовой акцент. Нельзя: текст, контент, брендинг.

Правила для edit_prompt:
- Пиши на английском (модель лучше понимает инструкции на английском)
- Описывай КОНКРЕТНЫЕ визуальные изменения
- Указывай позиции ("bottom-right", "top-left", "center")
- ОБЯЗАТЕЛЬНО добавь: "CRITICAL: Keep ALL existing text EXACTLY as-is — same language, same words, same font. Do NOT translate, replace, or remove any text. Do NOT add any new text, labels, badges, or icons."
- ОБЯЗАТЕЛЬНО добавь: "Preserve the brand logo, style, color palette, and overall composition."

Верни ТОЛЬКО JSON:
{{{{
    "edit_prompt": "Edit this advertising banner: ...",
    "preserve": ["all existing text exactly as-is", "brand logo", "color palette", "..."]
}}}}"""

    payload = {
        'model': 'gpt-5.2',
        'messages': [
            {
                'role': 'system',
                'content': 'You are a prompt engineer specializing in AI image editing instructions. You translate marketing recommendations into precise visual editing commands.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'max_completion_tokens': 1500,
        'reasoning_effort': 'medium'
    }

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        },
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        error_msg = f"GPT-5.2 API error: {response.status_code} - {response.text[:200]}"
        print(f"  ⚠️ {error_msg}")
        raise Exception(error_msg)

    result = response.json()
    text = result['choices'][0]['message']['content'].strip()

    # Parse JSON
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1])
        if text.startswith('json'):
            text = text[4:].strip()

    try:
        edit_data = json.loads(text)
        print(f"  ✅ Edit prompt built ({len(edit_data['edit_prompt'])} chars)")
        print(f"  Preserve: {edit_data.get('preserve', [])}")
        return edit_data
    except Exception as e:
        error_msg = f"Failed to parse GPT-5.2 response as JSON: {e}. Raw: {text[:300]}"
        print(f"  ⚠️ {error_msg}")
        raise Exception(error_msg)

# ============================================================================
# STEP 11: Regenerate Creative with GPT Image
# ============================================================================

def regenerate_creative(image_path, edit_data, output_path):
    """Regenerate creative using GPT Image edit mode"""
    print("  Regenerating creative with GPT Image...")

    edit_prompt = edit_data['edit_prompt']

    # Add preserve instructions to prompt
    preserve = edit_data.get('preserve', [])
    if preserve:
        edit_prompt += f"\n\nIMPORTANT: Preserve these elements unchanged: {', '.join(preserve)}."

    # Always add text preservation instruction
    edit_prompt += "\n\nCRITICAL RULE: Keep ALL existing text EXACTLY as-is in its original language. Do NOT translate, rewrite, remove, or replace any text on the banner. Do NOT add any new text, labels, badges, certificates, or icons that were not in the original image."

    # Determine best size for GPT Image based on original aspect ratio
    img = Image.open(image_path)
    orig_width, orig_height = img.size
    aspect = orig_width / orig_height

    # GPT Image supported sizes
    if aspect > 1.3:
        size = "1536x1024"  # landscape
    elif aspect < 0.77:
        size = "1024x1536"  # portrait
    else:
        size = "1024x1024"  # square-ish

    # Convert image to PNG for API (required format)
    png_buffer = io.BytesIO()
    img.convert('RGB').save(png_buffer, format='PNG')
    png_buffer.seek(0)
    img.close()

    # API call with retry
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'https://api.openai.com/v1/images/edits',
                headers={
                    'Authorization': f'Bearer {API_KEY}'
                },
                files={
                    'image': ('image.png', png_buffer, 'image/png')
                },
                data={
                    'model': 'gpt-image-1',
                    'prompt': edit_prompt,
                    'size': size,
                    'quality': 'high'
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                # GPT Image returns base64 encoded image
                image_b64 = result['data'][0]['b64_json']
                image_bytes = base64.b64decode(image_b64)

                # Save and resize back to original dimensions
                improved_img = Image.open(io.BytesIO(image_bytes))
                improved_img = improved_img.resize((orig_width, orig_height), Image.LANCZOS)

                # Convert to RGB and save as JPG
                if improved_img.mode == 'RGBA':
                    improved_img = improved_img.convert('RGB')
                improved_img.save(output_path, quality=95)
                improved_img.close()

                print(f"  ✅ Saved improved creative to: {output_path}")
                return output_path

            elif response.status_code == 400 and 'content_policy' in response.text.lower():
                print(f"  ⚠️ Content policy rejection — cannot regenerate this image")
                return None
            elif response.status_code == 429 or 'billing' in response.text.lower():
                print(f"  ⚠️ Rate limit or billing error: {response.text}")
                return None
            else:
                print(f"  ⚠️ GPT Image error (attempt {attempt+1}): {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    png_buffer.seek(0)
                    continue
                return None

        except requests.exceptions.Timeout:
            print(f"  ⚠️ Timeout (attempt {attempt+1})")
            if attempt < max_retries - 1:
                png_buffer.seek(0)
                continue
            return None

    return None

# ============================================================================
# STEP 9: Create Visualization
# ============================================================================

def create_visualization(image_path, zones, output_path):
    """Create clean visualization with legend panel"""
    print("  Creating visualization...")

    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Create extended canvas with legend panel on the right
    legend_width = 180
    new_width = img_width + legend_width
    new_img = Image.new('RGB', (new_width, img_height), (255, 255, 255))
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)

    try:
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except:
        font_small = ImageFont.load_default()
        font_title = font_small

    colors = {
        'logo': (255, 200, 0),       # Yellow
        'header': (0, 180, 80),      # Green
        'subheader': (255, 140, 0),  # Orange
        'cta': (80, 130, 255),       # Blue
        'product': (180, 80, 255),   # Purple
        'person': (255, 100, 150),   # Pink
        'slogan': (80, 180, 180),    # Cyan
        'description': (150, 150, 150),
        'legal': (120, 120, 120),
        'visual': (255, 100, 150)
    }

    # Sort zones by attention (highest first)
    zones_sorted = sorted(zones, key=lambda z: z.get('attention_pct', 0), reverse=True)

    # Draw legend panel header
    legend_x = img_width + 10
    legend_y = 15
    draw.text((legend_x, legend_y), "Зоны внимания", fill=(0, 0, 0), font=font_title)
    legend_y += 25

    # Draw zones on image (just thin borders, no labels)
    for i, zone in enumerate(zones_sorted):
        zone_type = zone['type']
        attention = zone.get('attention_pct', 0)
        x, y, w, h = zone['bbox']
        color = colors.get(zone_type, (200, 200, 200))

        # Draw thin rectangle border
        draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
        
        # Draw small number badge in corner
        badge_size = 16
        badge_x = x + 2
        badge_y = y + 2
        draw.ellipse([badge_x, badge_y, badge_x + badge_size, badge_y + badge_size], fill=color, outline=(255,255,255))
        
        # Number text (centered in badge)
        num_text = str(i + 1)
        num_bbox = draw.textbbox((0, 0), num_text, font=font_small)
        num_w = num_bbox[2] - num_bbox[0]
        num_h = num_bbox[3] - num_bbox[1]
        draw.text((badge_x + (badge_size - num_w) // 2, badge_y + (badge_size - num_h) // 2 - 1), 
                  num_text, fill=(0, 0, 0), font=font_small)

        # Add to legend
        # Color square
        draw.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12], fill=color, outline=(100,100,100))
        # Number
        draw.text((legend_x + 16, legend_y - 1), f"{i+1}.", fill=(0, 0, 0), font=font_small)
        # Type and attention
        legend_text = f"{zone_type} ({attention:.1f}%)"
        draw.text((legend_x + 32, legend_y - 1), legend_text, fill=(50, 50, 50), font=font_small)
        legend_y += 20
        
        # Add separator line every 3 items
        if (i + 1) % 3 == 0 and i < len(zones_sorted) - 1:
            legend_y += 5

    # Draw total coverage at bottom of legend
    total_attention = sum(z.get('attention_pct', 0) for z in zones)
    legend_y += 15
    draw.line([(legend_x, legend_y), (legend_x + legend_width - 20, legend_y)], fill=(200, 200, 200), width=1)
    legend_y += 10
    draw.text((legend_x, legend_y), f"Покрытие: {total_attention:.1f}%", fill=(0, 0, 0), font=font_title)

    new_img.save(output_path, quality=95)
    print(f"  ✅ Saved visualization to: {output_path}")

def save_heatmap(image_path, saliency_map, output_path):
    """Save saliency heatmap overlay"""
    print("  Creating heatmap...")

    # Load original image and convert to RGB
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img_array = np.array(img)

    # Normalize saliency map to 0-1
    saliency_normalized = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Apply colormap (hot)
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    heatmap_colored = cm.hot(saliency_normalized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend with original image
    alpha = 0.5
    blended = (alpha * heatmap_colored + (1 - alpha) * img_array).astype(np.uint8)

    # Save
    heatmap_img = Image.fromarray(blended)
    if heatmap_img.mode == 'RGBA':
        heatmap_img = heatmap_img.convert('RGB')

    heatmap_img.save(output_path, quality=95)
    print(f"  ✅ Saved heatmap to: {output_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def analyze_creative_final(image_path, filter_legal=True, regenerate=False):
    """Final complete analysis pipeline"""

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print("\n" + "="*70)
    print(f"FINAL CREATIVE ANALYSIS: {base_name}")
    print("="*70)

    # Get image dimensions without keeping full image in memory
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    print(f"Image size: {img_width}x{img_height}")

    # Step 1: Generate Saliency Map
    print_step(1, "Generate Saliency Map (DeepGaze)")
    img_array, saliency_map = generate_saliency_map(image_path)

    # Step 2: Detect Text Blocks
    print_step(2, "Detect Text Blocks (EasyOCR)")
    text_blocks = detect_text_blocks(image_path)

    # Step 3: Group and Classify Text Zones
    print_step(3, "Group and Classify Text Zones")
    text_zones = group_and_classify_text_zones(text_blocks, img_width, img_height)

    # Step 4: Detect Visual Elements
    print_step(4, "Detect Visual Elements (GPT-4.1)")
    visual_zones = detect_visual_elements_gpt41(image_path, text_zones, img_width, img_height)

    # Step 5: Refine CTA bbox
    print_step(5, "Refine CTA Bbox")
    cta_zone = next((z for z in text_zones if z['type'] == 'cta'), None)
    if cta_zone:
        refined_cta = refine_cta_bbox(image_path, cta_zone)
        # Update in text_zones
        for i, z in enumerate(text_zones):
            if z['type'] == 'cta':
                text_zones[i] = refined_cta
                break

    # Step 6: Merge All Zones
    print_step(6, "Merge All Zones")
    all_zones = merge_all_zones(text_zones, visual_zones, filter_legal)

    # Step 7: Calculate Attention
    print_step(7, "Calculate Attention per Zone")
    zones_with_attention, total_zones_attention = calculate_attention(saliency_map, all_zones)
    background_attention = 100 - total_zones_attention

    # Step 8: Generate Recommendations
    print_step(8, "Generate Recommendations (GPT-5.2)")
    recommendations = generate_recommendations(
        zones_with_attention,
        total_zones_attention,
        background_attention
    )

    # Step 9: Create Visualization
    print_step(9, "Create Visualization")
    viz_path = f"{base_name}_final.jpg"
    create_visualization(image_path, zones_with_attention, viz_path)

    # Save Heatmap
    heatmap_path = f"{base_name}_heatmap.jpg"
    save_heatmap(image_path, saliency_map, heatmap_path)

    # Step 10-11: Regenerate (optional)
    improved_path = None
    if regenerate:
        print_step(10, "Build Edit Prompt (GPT-5.2)")
        edit_data = build_edit_prompt(
            zones_with_attention,
            recommendations['recommendations'],
            img_width, img_height
        )

        if edit_data:
            print_step(11, "Regenerate Creative (GPT Image)")
            improved_path = regenerate_creative(
                image_path,
                edit_data,
                f"{base_name}_improved.jpg"
            )

    # Save Results
    save_step = 12 if regenerate else 10
    print_step(save_step, "Save Results")
    results = {
        'image': image_path,
        'zones': zones_with_attention,
        'total_zones_attention': round(total_zones_attention, 1),
        'background_attention': round(background_attention, 1),
        'overall_score': recommendations['overall_score'],
        'reasoning': recommendations['reasoning'],
        'recommendations': recommendations['recommendations'],
        'improved_image': improved_path
    }

    output_path = f"{base_name}_final.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved results to: {output_path}")

    # Print Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n📊 Overall Score: {recommendations['overall_score']}/5.0")
    print(f"\n💡 {recommendations['reasoning']}")

    print(f"\n📈 Zone Attention Distribution:\n")
    for zone in zones_with_attention:
        bar_length = int(zone['attention_pct'] * 2)
        bar = '█' * bar_length
        print(f"  [{zone['type']:12}] {zone['label'][:30]:30} | {zone['attention_pct']:5.1f}% {bar}")

    print(f"\n🎯 Top {min(3, len(recommendations['recommendations']))} Recommendations:\n")

    for i, rec in enumerate(recommendations['recommendations'][:3], 1):
        priority_emoji = {
            'High': '🔴',
            'Medium': '🟡',
            'Low': '🟢'
        }.get(rec['priority'], '⚪')

        print(f"{i}. {priority_emoji} [{rec['priority']}] {rec['title']}")
        print(f"   {rec['description'][:100]}...")
        print()

    print("="*70)
    print("✅ Analysis completed!")
    print(f"📁 Results: {output_path}")
    print(f"🖼️  Visualization: {viz_path}")
    if improved_path:
        print(f"🎨 Improved: {improved_path}")
    print("="*70)

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze advertising creatives')
    parser.add_argument('image_path', help='Path to the creative image')
    parser.add_argument('--regenerate', action='store_true',
                        help='Generate improved version of the banner')

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image not found: {args.image_path}")
        sys.exit(1)

    try:
        analyze_creative_final(args.image_path, regenerate=args.regenerate)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
