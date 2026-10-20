import os
from PIL import Image, ImageChops, ImageDraw, ImageFont

def trim(im, padding=20):
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        bbox = (max(0, bbox[0]-padding), max(0, bbox[1]-padding), 
                min(im.size[0], bbox[2]+padding), min(im.size[1], bbox[3]+padding))
        return im.crop(bbox)
    return im

datasets = [
    'h_b', 'h_k', 'h_l',
    'm_b', 'm_h', 'm_k',
    'r_b', 'r_k', 'r_l'
]

base_dir = 'result_inter/confidence/pinn'

try:
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf', 100)
except:
    try:
         font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 100)
    except:
        font = ImageFont.load_default()

def stitch():
    images = []
    for ds in datasets:
        path = os.path.join(base_dir, ds, "confidence_distribution.png")
        img = Image.open(path).convert('RGB')
        img = trim(img, padding=40)
        images.append(img)
    
    cols = 3
    rows = 3
    
    max_w = max(img.size[0] for img in images)
    max_h = max(img.size[1] for img in images)
    
    row_gap = 80 
    col_gap = 50 
    
    cell_w = max_w + col_gap
    cell_h = max_h + row_gap
    
    grid = Image.new('RGB', (cell_w * cols, cell_h * rows), color=(255,255,255))
    draw = ImageDraw.Draw(grid)
    
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        
        x_offset = c * cell_w + (cell_w - img.size[0]) // 2
        y_offset = r * cell_h + (row_gap // 2)
        
        grid.paste(img, (x_offset, y_offset))
        
        draw.text((x_offset + 30, y_offset + 10), letters[i], fill=(0,0,0), font=font)
        
    out_path = f"paper/confidence_combined.pdf"
    grid.save(out_path, "PDF", resolution=300.0)
    print(f"Saved {out_path}")

stitch()
