"""
Final automated creative analysis pipeline
Correct approach: OCR + GPT-4.1 visual + Saliency + GPT-5.2 recommendations

Usage: python3 analyze_creative_final.py <image_path>
"""

import sys
import os
import json
import base64
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

    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    model = DeepGazeIIE(pretrained=True)
    image_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)[None, ...]).float()

    centerbias = np.zeros((height, width))
    centerbias -= logsumexp(centerbias)
    centerbias_tensor = torch.from_numpy(centerbias[None, ...]).float()

    with torch.no_grad():
        log_density = model(image_tensor, centerbias_tensor)

    saliency_map = log_density.exp().numpy()[0, 0]

    print(f"  ✅ Saliency map generated ({width}x{height})")
    return img_array, saliency_map

# ============================================================================
# STEP 2: Detect Text Elements with OCR
# ============================================================================

def detect_text_blocks(image_path):
    """Detect text blocks using EasyOCR"""
    print("  Detecting text blocks with EasyOCR...")

    reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
    img = Image.open(image_path)
    img_array = np.array(img)

    results = reader.readtext(img_array)

    text_blocks = []
    for bbox, text, conf in results:
        # Get bounding box coordinates
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]

        x = int(min(x_coords))
        y = int(min(y_coords))
        w = int(max(x_coords) - x)
        h = int(max(y_coords) - y)

        # Skip low confidence or very small text
        if conf < 0.3 or len(text.strip()) < 2:
            continue

        text_blocks.append({
            'text': text.strip(),
            'bbox': [x, y, w, h],
            'confidence': float(conf)
        })

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
    """Detect visual elements (logo, product) using GPT-4.1"""
    print("  Detecting visual elements with GPT-4.1...")

    with open(image_path, 'rb') as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    image_format = "png" if image_path.endswith('.png') else "jpeg"

    # Describe existing zones to avoid duplication
    existing_desc = "\n".join([f"- {z['type']}: {z['bbox']}" for z in existing_zones])

    prompt = f"""Проанализируй рекламный креатив и найди ТОЛЬКО визуальные элементы.

Размер изображения: {img_width}x{img_height} пикселей

УЖЕ НАЙДЕННЫЕ текстовые зоны (НЕ детектируй их повторно):
{existing_desc}

Найди ТОЛЬКО:
1. "logo" - логотип бренда (иконка, небольшое изображение с названием)
2. "product" - изображение продукта/приложения/товара

ВАЖНО:
- bbox формат: [x, y, width, height]
- НЕ включай текстовые элементы
- НЕ включай декоративные элементы (фоновые фигуры, лучи)

Верни ТОЛЬКО JSON массив (2-3 элемента максимум):
[
  {{"type": "logo", "label": "название бренда", "bbox": [x, y, width, height]}},
  {{"type": "product", "label": "описание продукта", "bbox": [x, y, width, height]}}
]"""

    payload = {
        'model': 'gpt-4.1',
        'messages': [
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
        'max_tokens': 1000
    }

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        },
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        print(f"  ⚠️  GPT-4.1 error: {response.status_code}")
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
        print(f"  ✅ Found {len(visual_zones)} visual elements")
        return visual_zones
    except:
        print(f"  ⚠️  Failed to parse GPT-4.1 response")
        return []

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

    prompt = f"""Ты эксперт по анализу рекламных креативов и eye-tracking данным.

Проанализируй результаты eye-tracking анализа **рекламного баннера**:

**Данные анализа зон:**
{json.dumps(zones_summary, indent=2, ensure_ascii=False)}

**Покрытие:**
- Зоны: {total_zones_attention}%
- Фон: {background_attention}%

**КОНТЕКСТ:**
Это рекламный баннер, который сам может быть кликабельным. Призыв к действию может быть:
- В виде отдельной CTA кнопки (оптимально 15-25% внимания)
- В самом заголовке/оффере (например "КУПИТЕ СЕЙЧАС" как header)
- Имплицитным (весь баннер кликабелен)

**Твоя задача:**

1. **Overall Score (1-5)** - общая оценка эффективности
2. **Reasoning** - краткое объяснение (2-3 предложения)
3. **Рекомендации (3-5 штук)**:
   - priority: "High" / "Medium" / "Low"
   - title: короткий заголовок
   - description: подробное описание
   - expected_impact: ожидаемый эффект

**ВАЖНО:**
- Если CTA кнопка есть - она должна получать 15-25% внимания
- Если CTA кнопки нет - проверь, есть ли призыв в заголовке
- Legal text (1-3%) - норма
- Фон >20% - проблема (отвлекает от контента)
- Для баннеров важнее clarity оффера, чем наличие кнопки

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
                'content': 'Ты эксперт по рекламным креативам и eye-tracking анализу с 10+ летним опытом.'
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
# STEP 9: Create Visualization
# ============================================================================

def create_visualization(image_path, zones, output_path):
    """Create visualization with bounding boxes"""
    print("  Creating visualization...")

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    try:
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font_small = ImageFont.load_default()

    colors = {
        'logo': (255, 200, 0),
        'header': (0, 200, 100),
        'subheader': (255, 120, 0),
        'cta': (100, 150, 255),
        'product': (200, 100, 255),
        'description': (150, 150, 150),
        'legal': (100, 100, 100),
        'visual': (255, 100, 150)
    }

    for zone in zones:
        zone_type = zone['type']
        attention = zone.get('attention_pct', 0)
        x, y, w, h = zone['bbox']

        color = colors.get(zone_type, (255, 255, 255))

        # Draw rectangle
        draw.rectangle([x, y, x+w, y+h], outline=color, width=3)

        # Draw label
        label_text = f"{zone_type}: {attention:.1f}%"
        text_bbox = draw.textbbox((x+3, y-22), label_text, font=font_small)
        text_bbox = (text_bbox[0]-3, text_bbox[1]-2, text_bbox[2]+3, text_bbox[3]+2)

        draw.rectangle(text_bbox, fill=color)
        draw.text((x+3, y-22), label_text, fill=(0, 0, 0), font=font_small)

    if img.mode == 'RGBA':
        img = img.convert('RGB')

    img.save(output_path, quality=95)
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

def analyze_creative_final(image_path, filter_legal=True):
    """Final complete analysis pipeline"""

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print("\n" + "="*70)
    print(f"FINAL CREATIVE ANALYSIS: {base_name}")
    print("="*70)

    # Get image dimensions
    img = Image.open(image_path)
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

    # Save Results
    print_step(10, "Save Results")
    results = {
        'image': image_path,
        'zones': zones_with_attention,
        'total_zones_attention': round(total_zones_attention, 1),
        'background_attention': round(background_attention, 1),
        'overall_score': recommendations['overall_score'],
        'reasoning': recommendations['reasoning'],
        'recommendations': recommendations['recommendations']
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
    print("="*70)

    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_creative_final.py <image_path>")
        print("\nExample:")
        print("  python3 analyze_creative_final.py yandex_pay.png")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)

    try:
        analyze_creative_final(image_path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
