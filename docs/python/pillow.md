# Pillow (PIL)

Pillow is the modern, friendly fork of the Python Imaging Library (PIL). It provides extensive file format support, efficient internal representation, and powerful image processing capabilities. Pillow is the de facto standard for image manipulation in Python.

## Installation

```bash
# Basic installation
pip install Pillow

# With optional dependencies
pip install Pillow[complete]  # Includes all optional dependencies

# With specific extras
pip install "Pillow[extra]"   # WebP, PDF, and other format support

# Development version
pip install git+https://github.com/python-pillow/Pillow.git

# System dependencies (Ubuntu/Debian)
sudo apt-get install libjpeg-dev zlib1g-dev libtiff-dev libfreetype6-dev
```

## Basic Setup

```python
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from PIL import ImageOps, ImageChops, ImageStat, ImageColor
import os
import io
import numpy as np

# Check Pillow version and features
print(f"Pillow version: {Image.__version__}")

# Check supported formats
print("Supported formats:")
print("Read:", ", ".join(sorted(Image.registered_extensions())))
print("Write:", ", ".join(sorted(Image.SAVE.keys())))
```

## Core Functionality

### Loading and Saving Images

```python
# Load images
img = Image.open("path/to/image.jpg")
img = Image.open("path/to/image.png")

# Load from URL
import requests
from io import BytesIO
response = requests.get("https://example.com/image.jpg")
img = Image.open(BytesIO(response.content))

# Load from bytes
with open("image.jpg", "rb") as f:
    img = Image.open(BytesIO(f.read()))

# Basic image information
print(f"Format: {img.format}")        # JPEG, PNG, etc.
print(f"Mode: {img.mode}")           # RGB, RGBA, L, etc.
print(f"Size: {img.size}")           # (width, height)
print(f"Info: {img.info}")           # Metadata dictionary

# Save images
img.save("output.jpg")                # Auto-detect format from extension
img.save("output.png", "PNG")         # Explicit format
img.save("output.jpg", quality=95)    # JPEG with quality setting

# Save with optimization
img.save("optimized.jpg", optimize=True, quality=85)
img.save("progressive.jpg", progressive=True, quality=90)

# Save to bytes
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
image_bytes = buffer.getvalue()
```

### Image Modes and Conversions

```python
# Common image modes
# L: Grayscale (8-bit)
# RGB: Color (3x8-bit)
# RGBA: Color with transparency (4x8-bit)
# CMYK: Color for printing (4x8-bit)
# P: Palette mode (8-bit mapped)

# Mode conversions
rgb_img = img.convert('RGB')          # Convert to RGB
gray_img = img.convert('L')           # Convert to grayscale
rgba_img = img.convert('RGBA')        # Add alpha channel

# Grayscale with custom weights
def rgb_to_gray_custom(img, weights=(0.299, 0.587, 0.114)):
    """Convert RGB to grayscale with custom weights"""
    r, g, b = img.split()
    gray = Image.eval(r, lambda x: int(x * weights[0])) 
    gray.paste(Image.eval(g, lambda x: int(x * weights[1])), mask=None)
    gray.paste(Image.eval(b, lambda x: int(x * weights[2])), mask=None)
    return gray

# Color quantization (reduce colors)
quantized = img.quantize(colors=256)   # Reduce to 256 colors
quantized = img.quantize(colors=8)     # Reduce to 8 colors
```

### Basic Image Operations

```python
# Resize images
resized = img.resize((800, 600))              # Resize to specific size
resized = img.resize((400, 300), Image.LANCZOS)  # High-quality resampling

# Maintain aspect ratio
def resize_with_aspect(img, max_size=(800, 600)):
    img.thumbnail(max_size, Image.LANCZOS)    # In-place resize maintaining aspect
    return img

# Crop images
cropped = img.crop((100, 100, 400, 400))      # (left, top, right, bottom)

# Rotate images
rotated = img.rotate(45)                      # Rotate 45 degrees
rotated = img.rotate(30, expand=True)         # Expand canvas to fit
rotated = img.rotate(-90, fillcolor='white')  # Fill empty areas with white

# Flip and transpose
flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)   # Horizontal flip
flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)   # Vertical flip
rotated_90 = img.transpose(Image.ROTATE_90)        # 90-degree rotation
rotated_180 = img.transpose(Image.ROTATE_180)      # 180-degree rotation
rotated_270 = img.transpose(Image.ROTATE_270)      # 270-degree rotation

# Paste one image onto another
background = Image.new('RGB', (800, 600), 'white')
background.paste(img, (100, 100))            # Paste at position
background.paste(img, (200, 200), img)       # Use img as mask (if RGBA)
```

## Common Use Cases

### Image Resizing and Thumbnails

```python
def create_thumbnail(input_path, output_path, size=(150, 150)):
    """Create a thumbnail maintaining aspect ratio"""
    with Image.open(input_path) as img:
        # Convert to RGB if necessary (for JPEG output)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        img.thumbnail(size, Image.LANCZOS)
        img.save(output_path, 'JPEG', quality=85, optimize=True)

def resize_to_fit(img, target_size, background_color='white'):
    """Resize image to fit within target size, adding padding if needed"""
    img.thumbnail(target_size, Image.LANCZOS)
    
    # Create new image with target size
    new_img = Image.new('RGB', target_size, background_color)
    
    # Calculate position to center the image
    x = (target_size[0] - img.width) // 2
    y = (target_size[1] - img.height) // 2
    
    new_img.paste(img, (x, y))
    return new_img

def resize_to_cover(img, target_size):
    """Resize and crop image to cover target size"""
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        # Image is wider, resize by height
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Crop horizontally
        left = (new_width - target_size[0]) // 2
        img = img.crop((left, 0, left + target_size[0], target_size[1]))
    else:
        # Image is taller, resize by width
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Crop vertically
        top = (new_height - target_size[1]) // 2
        img = img.crop((0, top, target_size[0], top + target_size[1]))
    
    return img

# Usage examples
create_thumbnail('large_image.jpg', 'thumbnail.jpg', (200, 200))

img = Image.open('original.jpg')
fitted = resize_to_fit(img, (800, 600), 'black')
covered = resize_to_cover(img, (800, 600))
```

### Image Filters and Enhancement

```python
# Built-in filters
blurred = img.filter(ImageFilter.BLUR)
sharp = img.filter(ImageFilter.SHARPEN)
smooth = img.filter(ImageFilter.SMOOTH)
detail = img.filter(ImageFilter.DETAIL)
edge_enhance = img.filter(ImageFilter.EDGE_ENHANCE)
emboss = img.filter(ImageFilter.EMBOSS)
contour = img.filter(ImageFilter.CONTOUR)

# Gaussian blur with radius
gaussian = img.filter(ImageFilter.GaussianBlur(radius=2))

# Box blur
box_blur = img.filter(ImageFilter.BoxBlur(radius=1))

# Unsharp mask for sharpening
unsharp = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

# Kernel-based filters
kernel_sharpen = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[-1, -1, -1, -1, 9, -1, -1, -1, -1],
    scale=1
)
sharpened = img.filter(kernel_sharpen)

# Edge detection kernel
kernel_edge = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
    scale=1
)
edges = img.filter(kernel_edge)

# Image enhancement
enhancer = ImageEnhance.Brightness(img)
brighter = enhancer.enhance(1.3)      # 30% brighter
darker = enhancer.enhance(0.7)        # 30% darker

enhancer = ImageEnhance.Contrast(img)
high_contrast = enhancer.enhance(1.5) # Increase contrast

enhancer = ImageEnhance.Color(img)
saturated = enhancer.enhance(1.4)     # More saturated
desaturated = enhancer.enhance(0.6)   # Less saturated

enhancer = ImageEnhance.Sharpness(img)
sharp = enhancer.enhance(2.0)         # Sharper
soft = enhancer.enhance(0.5)          # Softer
```

### Drawing on Images

```python
def add_watermark(img, text, position='bottom-right', font_size=36, opacity=128):
    """Add text watermark to image"""
    # Create a transparent overlay
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    margin = 20
    if position == 'bottom-right':
        x = img.width - text_width - margin
        y = img.height - text_height - margin
    elif position == 'bottom-left':
        x = margin
        y = img.height - text_height - margin
    elif position == 'top-right':
        x = img.width - text_width - margin
        y = margin
    else:  # top-left
        x = margin
        y = margin
    
    # Draw text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity))
    
    # Composite with original image
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    watermarked = Image.alpha_composite(img, overlay)
    return watermarked.convert('RGB')

def draw_shapes_and_text(img):
    """Draw various shapes and text on image"""
    draw = ImageDraw.Draw(img)
    
    # Rectangle
    draw.rectangle([50, 50, 150, 100], fill='red', outline='black', width=2)
    
    # Circle (ellipse with equal width and height)
    draw.ellipse([200, 50, 300, 150], fill='blue', outline='white', width=3)
    
    # Line
    draw.line([50, 150, 300, 200], fill='green', width=5)
    
    # Polygon
    draw.polygon([(400, 50), (450, 100), (400, 150), (350, 100)], 
                fill='yellow', outline='purple')
    
    # Text
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 220), "Hello, Pillow!", font=font, fill='black')
    
    return img

def add_border(img, border_size=10, color='black'):
    """Add border around image"""
    return ImageOps.expand(img, border=border_size, fill=color)

def create_rounded_corners(img, radius=20):
    """Create image with rounded corners"""
    # Create mask
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, img.width, img.height], radius, fill=255)
    
    # Apply mask
    img = img.convert('RGBA')
    img.putalpha(mask)
    
    return img

# Usage examples
img = Image.open('photo.jpg')
watermarked = add_watermark(img, "© 2025 My Company", opacity=100)
bordered = add_border(img, 15, 'white')
rounded = create_rounded_corners(img, 30)
```

### Batch Processing

```python
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def process_image(input_path, output_path, operations):
    """Process a single image with specified operations"""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Apply operations
            for operation, params in operations.items():
                if operation == 'resize':
                    img = img.resize(params['size'], Image.LANCZOS)
                elif operation == 'thumbnail':
                    img.thumbnail(params['size'], Image.LANCZOS)
                elif operation == 'rotate':
                    img = img.rotate(params['angle'])
                elif operation == 'enhance_brightness':
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(params['factor'])
                elif operation == 'enhance_contrast':
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(params['factor'])
                elif operation == 'filter':
                    img = img.filter(params['filter'])
                elif operation == 'grayscale':
                    img = img.convert('L')
            
            # Save processed image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path, quality=params.get('quality', 90), optimize=True)
            
        return f"Processed: {input_path} -> {output_path}"
        
    except Exception as e:
        return f"Error processing {input_path}: {str(e)}"

def batch_process_images(input_dir, output_dir, operations, file_extensions=None):
    """Batch process images in a directory"""
    if file_extensions is None:
        file_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all image files
    image_files = []
    for ext in file_extensions:
        image_files.extend(input_path.glob(f"**/*{ext}"))
        image_files.extend(input_path.glob(f"**/*{ext.upper()}"))
    
    # Prepare tasks
    tasks = []
    for img_file in image_files:
        relative_path = img_file.relative_to(input_path)
        output_file = output_path / relative_path
        tasks.append((str(img_file), str(output_file), operations))
    
    # Process with multiple threads
    max_workers = min(32, multiprocessing.cpu_count() * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda args: process_image(*args), tasks)
    
    # Print results
    for result in results:
        print(result)

# Usage example
operations = {
    'thumbnail': {'size': (800, 600)},
    'enhance_contrast': {'factor': 1.1},
    'filter': {'filter': ImageFilter.SHARPEN},
    'quality': 85
}

batch_process_images('./input_photos', './output_photos', operations)

# Specific batch operations
def create_thumbnails_batch(input_dir, output_dir, size=(200, 200)):
    """Create thumbnails for all images in directory"""
    operations = {
        'thumbnail': {'size': size},
        'quality': 85
    }
    batch_process_images(input_dir, output_dir, operations)

def optimize_images_batch(input_dir, output_dir, quality=85):
    """Optimize images for web (reduce file size)"""
    operations = {
        'resize': {'size': (1920, 1080)},  # Max size for web
        'quality': quality
    }
    batch_process_images(input_dir, output_dir, operations)

# Usage
create_thumbnails_batch('./photos', './thumbnails', (150, 150))
optimize_images_batch('./large_photos', './optimized', quality=75)
```

## Advanced Features

### Working with Image Metadata

```python
from PIL.ExifTags import TAGS, GPSTAGS
import json

def extract_exif_data(img_path):
    """Extract EXIF data from image"""
    with Image.open(img_path) as img:
        exif_data = {}
        
        # Get raw EXIF data
        exif = img._getexif()
        
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                
                # Handle GPS data specially
                if tag == "GPSInfo":
                    gps_data = {}
                    for gps_tag_id, gps_value in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag] = gps_value
                    exif_data[tag] = gps_data
                else:
                    exif_data[tag] = value
        
        return exif_data

def get_image_info(img_path):
    """Get comprehensive image information"""
    with Image.open(img_path) as img:
        info = {
            'filename': os.path.basename(img_path),
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'width': img.width,
            'height': img.height,
            'aspect_ratio': round(img.width / img.height, 2),
            'file_size': os.path.getsize(img_path),
            'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
        }
        
        # Add EXIF data if available
        exif = extract_exif_data(img_path)
        if exif:
            info['exif'] = {
                'make': exif.get('Make', 'Unknown'),
                'model': exif.get('Model', 'Unknown'),
                'datetime': exif.get('DateTime', 'Unknown'),
                'orientation': exif.get('Orientation', 1),
                'iso': exif.get('ISOSpeedRatings', 'Unknown'),
                'focal_length': exif.get('FocalLength', 'Unknown')
            }
        
        return info

def auto_rotate_image(img_path, output_path=None):
    """Auto-rotate image based on EXIF orientation"""
    with Image.open(img_path) as img:
        # Get orientation from EXIF
        exif = img._getexif()
        orientation = 1
        
        if exif and 'Orientation' in [TAGS.get(k) for k in exif.keys()]:
            for tag_id, value in exif.items():
                if TAGS.get(tag_id) == 'Orientation':
                    orientation = value
                    break
        
        # Apply rotation based on orientation
        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
        
        # Save rotated image
        if output_path:
            img.save(output_path)
        else:
            img.save(img_path)
        
        return img

# Usage
info = get_image_info('photo.jpg')
print(json.dumps(info, indent=2, default=str))

auto_rotate_image('rotated_photo.jpg', 'corrected_photo.jpg')
```

### Color Operations

```python
def adjust_color_balance(img, cyan_red=0, magenta_green=0, yellow_blue=0):
    """Adjust color balance similar to Photoshop"""
    # Split into RGB channels
    r, g, b = img.split()
    
    # Apply adjustments
    r = r.point(lambda x: max(0, min(255, x + cyan_red)))
    g = g.point(lambda x: max(0, min(255, x + magenta_green)))
    b = b.point(lambda x: max(0, min(255, x + yellow_blue)))
    
    return Image.merge('RGB', (r, g, b))

def apply_color_curves(img, curve_points):
    """Apply color curves adjustment"""
    # Create lookup table
    curve = list(range(256))
    
    # Interpolate curve points
    for i in range(len(curve_points) - 1):
        x1, y1 = curve_points[i]
        x2, y2 = curve_points[i + 1]
        
        for x in range(x1, x2 + 1):
            if x2 > x1:
                ratio = (x - x1) / (x2 - x1)
                curve[x] = int(y1 + ratio * (y2 - y1))
    
    return img.point(curve)

def create_sepia_effect(img):
    """Create sepia tone effect"""
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Apply sepia transformation matrix
    pixels = img.load()
    width, height = img.size
    
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            
            # Sepia transformation
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            
            # Ensure values are within range
            pixels[x, y] = (min(255, tr), min(255, tg), min(255, tb))
    
    return img

def create_vintage_effect(img):
    """Create vintage photo effect"""
    # Apply sepia
    img = create_sepia_effect(img)
    
    # Reduce contrast slightly
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(0.9)
    
    # Add slight blur
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    
    # Add vignette effect
    width, height = img.size
    vignette = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(vignette)
    
    # Create radial gradient for vignette
    center_x, center_y = width // 2, height // 2
    max_distance = min(width, height) // 2
    
    for y in range(height):
        for x in range(width):
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            if distance < max_distance:
                alpha = int(255 * (1 - (distance / max_distance) ** 2))
                vignette.putpixel((x, y), alpha)
            else:
                vignette.putpixel((x, y), 0)
    
    # Apply vignette
    img = img.convert('RGBA')
    img.putalpha(vignette)
    
    return img

def extract_dominant_colors(img, num_colors=5):
    """Extract dominant colors from image"""
    # Convert to RGB and resize for performance
    img = img.convert('RGB')
    img = img.resize((150, 150))  # Smaller size for faster processing
    
    # Quantize to reduce colors
    quantized = img.quantize(colors=num_colors)
    
    # Get palette colors
    palette = quantized.getpalette()
    colors = []
    
    for i in range(num_colors):
        r = palette[i * 3]
        g = palette[i * 3 + 1]
        b = palette[i * 3 + 2]
        colors.append((r, g, b))
    
    # Count occurrences of each color
    quantized = quantized.convert('RGB')
    pixels = list(quantized.getdata())
    
    color_counts = {}
    for pixel in pixels:
        color_counts[pixel] = color_counts.get(pixel, 0) + 1
    
    # Sort by frequency
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    return [color for color, count in sorted_colors[:num_colors]]

# Usage examples
img = Image.open('photo.jpg')

# Color adjustments
balanced = adjust_color_balance(img, cyan_red=10, yellow_blue=-5)

# Curves adjustment (darken shadows, brighten highlights)
curves = apply_color_curves(img, [(0, 0), (64, 50), (128, 128), (192, 200), (255, 255)])

# Effects
sepia = create_sepia_effect(img)
vintage = create_vintage_effect(img)

# Extract colors
dominant_colors = extract_dominant_colors(img)
print("Dominant colors:", dominant_colors)
```

### Advanced Compositing

```python
def blend_images(img1, img2, mode='normal', opacity=0.5):
    """Blend two images with different blend modes"""
    # Ensure images are same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)
    
    # Convert to same mode
    if img1.mode != img2.mode:
        img2 = img2.convert(img1.mode)
    
    if mode == 'normal':
        return Image.blend(img1, img2, opacity)
    elif mode == 'multiply':
        return ImageChops.multiply(img1, img2)
    elif mode == 'screen':
        return ImageChops.screen(img1, img2)
    elif mode == 'overlay':
        return ImageChops.overlay(img1, img2)
    elif mode == 'difference':
        return ImageChops.difference(img1, img2)
    elif mode == 'add':
        return ImageChops.add(img1, img2)
    elif mode == 'subtract':
        return ImageChops.subtract(img1, img2)
    elif mode == 'darker':
        return ImageChops.darker(img1, img2)
    elif mode == 'lighter':
        return ImageChops.lighter(img1, img2)
    else:
        return Image.blend(img1, img2, opacity)

def create_photo_collage(images, layout=(2, 2), spacing=10, background_color='white'):
    """Create a photo collage from multiple images"""
    rows, cols = layout
    
    # Calculate dimensions for each cell
    total_images = len(images)
    images = images[:rows * cols]  # Limit to layout size
    
    # Find maximum dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # Calculate canvas size
    canvas_width = cols * max_width + (cols + 1) * spacing
    canvas_height = rows * max_height + (rows + 1) * spacing
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), background_color)
    
    # Place images
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        # Resize image to fit cell
        img_resized = img.copy()
        img_resized.thumbnail((max_width, max_height), Image.LANCZOS)
        
        # Calculate position
        x = spacing + col * (max_width + spacing) + (max_width - img_resized.width) // 2
        y = spacing + row * (max_height + spacing) + (max_height - img_resized.height) // 2
        
        canvas.paste(img_resized, (x, y))
    
    return canvas

def apply_gradient_mask(img, direction='horizontal', start_alpha=255, end_alpha=0):
    """Apply gradient mask to image"""
    # Create gradient mask
    width, height = img.size
    mask = Image.new('L', (width, height))
    
    for y in range(height):
        for x in range(width):
            if direction == 'horizontal':
                alpha = int(start_alpha + (end_alpha - start_alpha) * x / width)
            elif direction == 'vertical':
                alpha = int(start_alpha + (end_alpha - start_alpha) * y / height)
            elif direction == 'radial':
                center_x, center_y = width // 2, height // 2
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                max_distance = min(width, height) // 2
                alpha = int(start_alpha + (end_alpha - start_alpha) * min(distance / max_distance, 1))
            
            mask.putpixel((x, y), max(0, min(255, alpha)))
    
    # Apply mask
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    img.putalpha(mask)
    return img

def create_panorama(images):
    """Simple panorama creation (basic horizontal stitching)"""
    if not images:
        return None
    
    # Resize all images to same height
    min_height = min(img.height for img in images)
    resized_images = []
    
    for img in images:
        aspect_ratio = img.width / img.height
        new_width = int(min_height * aspect_ratio)
        resized_img = img.resize((new_width, min_height), Image.LANCZOS)
        resized_images.append(resized_img)
    
    # Calculate total width
    total_width = sum(img.width for img in resized_images)
    
    # Create panorama canvas
    panorama = Image.new('RGB', (total_width, min_height))
    
    # Paste images horizontally
    x_offset = 0
    for img in resized_images:
        panorama.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return panorama

# Usage examples
img1 = Image.open('photo1.jpg')
img2 = Image.open('photo2.jpg')

# Blend images
blended = blend_images(img1, img2, 'overlay', 0.7)

# Create collage
images = [Image.open(f'photo{i}.jpg') for i in range(1, 5)]
collage = create_photo_collage(images, (2, 2), spacing=15)

# Apply gradient
gradient_img = apply_gradient_mask(img1, 'radial')

# Create panorama
pano_images = [Image.open(f'pano{i}.jpg') for i in range(1, 4)]
panorama = create_panorama(pano_images)
```

## Integration with Other Libraries

### With NumPy

```python
import numpy as np

# Convert PIL Image to NumPy array
img_array = np.array(img)
print(f"Array shape: {img_array.shape}")  # (height, width, channels)
print(f"Array dtype: {img_array.dtype}")  # uint8

# Convert NumPy array to PIL Image
img_from_array = Image.fromarray(img_array)

# Advanced NumPy operations
def adjust_gamma(img, gamma=1.0):
    """Apply gamma correction"""
    img_array = np.array(img, dtype=np.float64)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.power(img_array, gamma)
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def apply_histogram_equalization(img):
    """Apply histogram equalization using NumPy"""
    if img.mode != 'L':
        img = img.convert('L')
    
    img_array = np.array(img)
    
    # Calculate histogram
    hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
    
    # Calculate cumulative distribution
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # Apply equalization
    img_equalized = np.interp(img_array.flatten(), bins[:-1], cdf_normalized)
    img_equalized = img_equalized.reshape(img_array.shape).astype(np.uint8)
    
    return Image.fromarray(img_equalized)

def create_noise(size, noise_type='gaussian'):
    """Create noise using NumPy"""
    if noise_type == 'gaussian':
        noise = np.random.normal(128, 30, size).astype(np.uint8)
    elif noise_type == 'uniform':
        noise = np.random.uniform(0, 255, size).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        noise = np.random.choice([0, 255], size=size, p=[0.5, 0.5]).astype(np.uint8)
    
    return Image.fromarray(noise)

# Usage
gamma_corrected = adjust_gamma(img, gamma=1.5)
equalized = apply_histogram_equalization(img)
noise_img = create_noise((200, 200, 3), 'gaussian')
```

### With Matplotlib

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_image_analysis(img):
    """Create comprehensive image analysis plot"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Grayscale
    gray_img = img.convert('L')
    axes[0, 1].imshow(gray_img, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # Edge detection
    edges = gray_img.filter(ImageFilter.FIND_EDGES)
    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title('Edge Detection')
    axes[0, 2].axis('off')
    
    # RGB Histogram
    if img.mode == 'RGB':
        r_hist = np.array(img.split()[0]).flatten()
        g_hist = np.array(img.split()[1]).flatten()
        b_hist = np.array(img.split()[2]).flatten()
        
        axes[1, 0].hist(r_hist, bins=256, color='red', alpha=0.7, density=True)
        axes[1, 0].hist(g_hist, bins=256, color='green', alpha=0.7, density=True)
        axes[1, 0].hist(b_hist, bins=256, color='blue', alpha=0.7, density=True)
        axes[1, 0].set_title('RGB Histogram')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Density')
    
    # Grayscale histogram
    gray_hist = np.array(gray_img).flatten()
    axes[1, 1].hist(gray_hist, bins=256, color='gray', alpha=0.7)
    axes[1, 1].set_title('Grayscale Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # Image statistics
    stats = ImageStat.Stat(img)
    axes[1, 2].axis('off')
    stats_text = f"""
    Size: {img.size}
    Mode: {img.mode}
    Mean: {[round(m, 1) for m in stats.mean]}
    Median: {[round(m, 1) for m in stats.median]}
    StdDev: {[round(s, 1) for s in stats.stddev]}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    axes[1, 2].set_title('Image Statistics')
    
    plt.tight_layout()
    plt.show()

def create_before_after_plot(original, processed, title="Before / After"):
    """Create side-by-side comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title('Before')
    axes[0].axis('off')
    
    axes[1].imshow(processed)
    axes[1].set_title('After')
    axes[1].axis('off')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Usage
plot_image_analysis(img)
processed_img = img.filter(ImageFilter.SHARPEN)
create_before_after_plot(img, processed_img, "Sharpening Effect")
```

### With OpenCV Integration

```python
import cv2

def pil_to_opencv(pil_img):
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def opencv_to_pil(cv_img):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def advanced_edge_detection(img):
    """Advanced edge detection using OpenCV"""
    # Convert to OpenCV format
    cv_img = pil_to_opencv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Convert back to PIL
    return Image.fromarray(edges, mode='L')

def detect_contours(img, min_area=1000):
    """Detect and draw contours"""
    cv_img = pil_to_opencv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and draw
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(cv_img, [contour], -1, (0, 255, 0), 2)
    
    return opencv_to_pil(cv_img)

# Usage
edges = advanced_edge_detection(img)
contours = detect_contours(img)
```

## Best Practices

### Performance Optimization

```python
# 1. Use appropriate resampling filters
resizing_filters = {
    'fastest': Image.NEAREST,      # Fastest, lowest quality
    'balanced': Image.BILINEAR,    # Good balance
    'quality': Image.LANCZOS       # Best quality, slower
}

# 2. Work with smaller images when possible
def optimize_for_processing(img, max_size=1024):
    """Resize large images for faster processing"""
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img

# 3. Use context managers for file handling
def process_multiple_images(file_paths, operations):
    """Efficiently process multiple images"""
    results = []
    for path in file_paths:
        with Image.open(path) as img:
            # Process without loading full image into memory
            for op in operations:
                img = op(img)
            results.append(img.copy())  # Make a copy before context ends
    return results

# 4. Batch operations when possible
def batch_resize(image_paths, size, output_dir):
    """Batch resize operation"""
    os.makedirs(output_dir, exist_ok=True)
    
    for path in image_paths:
        with Image.open(path) as img:
            img.thumbnail(size, Image.LANCZOS)
            filename = os.path.basename(path)
            img.save(os.path.join(output_dir, filename), optimize=True)

# 5. Use appropriate image formats
format_recommendations = {
    'photos': 'JPEG',           # Best for photos
    'graphics': 'PNG',          # Best for graphics with transparency
    'web_photos': 'WebP',       # Modern web format
    'icons': 'PNG',             # Best for icons
    'print': 'TIFF'            # Best for print
}
```

### Memory Management

```python
import psutil
import gc

def monitor_memory_usage():
    """Monitor memory usage"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def process_large_image_safely(img_path, operations, chunk_size=None):
    """Process large images without loading everything into memory"""
    with Image.open(img_path) as img:
        print(f"Processing {img.size} image")
        print(f"Memory before: {monitor_memory_usage():.1f} MB")
        
        # Work with image
        for operation in operations:
            img = operation(img)
            gc.collect()  # Force garbage collection
        
        print(f"Memory after: {monitor_memory_usage():.1f} MB")
        return img

def create_image_pyramid(img, levels=3):
    """Create image pyramid to save memory"""
    pyramid = [img]
    current = img
    
    for level in range(1, levels):
        size = (current.width // 2, current.height // 2)
        current = current.resize(size, Image.LANCZOS)
        pyramid.append(current)
    
    return pyramid
```

### Error Handling and Validation

```python
def safe_image_operation(operation):
    """Decorator for safe image operations"""
    def wrapper(img_path, *args, **kwargs):
        try:
            with Image.open(img_path) as img:
                return operation(img, *args, **kwargs)
        except IOError:
            print(f"Cannot open image: {img_path}")
            return None
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None
    return wrapper

@safe_image_operation
def safe_resize(img, size):
    """Safely resize image with error handling"""
    return img.resize(size, Image.LANCZOS)

def validate_image_format(img_path, allowed_formats=None):
    """Validate image format"""
    if allowed_formats is None:
        allowed_formats = {'JPEG', 'PNG', 'BMP', 'TIFF', 'WebP'}
    
    try:
        with Image.open(img_path) as img:
            if img.format in allowed_formats:
                return True, img.format
            else:
                return False, f"Format {img.format} not allowed"
    except:
        return False, "Cannot open file"

def robust_image_save(img, output_path, fallback_format='JPEG', **kwargs):
    """Save image with fallback options"""
    try:
        img.save(output_path, **kwargs)
        return True, "Saved successfully"
    except:
        try:
            # Try with fallback format
            base_name = os.path.splitext(output_path)[0]
            fallback_path = f"{base_name}.{fallback_format.lower()}"
            
            if img.mode in ('RGBA', 'LA'):
                img = img.convert('RGB')
                
            img.save(fallback_path, fallback_format, quality=90)
            return True, f"Saved as {fallback_format}"
        except Exception as e:
            return False, str(e)

# Usage
result = safe_resize('photo.jpg', (800, 600))
valid, msg = validate_image_format('image.xyz')
success, msg = robust_image_save(img, 'output.webp')
```

This comprehensive cheat sheet covers the essential aspects of Pillow for image processing in Python. The library's strength lies in its broad format support, ease of use, and extensive functionality for both simple and complex image operations. It's the go-to library for Python developers working with images in web development, data science, and desktop applications.