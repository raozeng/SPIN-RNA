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
    'Atlas_h_m1A', 'Atlas_h_m6A', 
    'Atlas_m_m5C', 'Atlas_r_m6A', 
    'RMBase_h_PseudoU', 'Atlas_h_m7G'
]

base_dir = 'result_inter/features/pinn'

# Attempt to load a good serif font, size 100 for 300dpi readability
try:
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf', 100)
except:
    try:
         font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 100)
    except:
        font = ImageFont.load_default()

def stitch(prefix):
    images = []
    for ds in datasets:
        path = os.path.join(base_dir, ds, f"{prefix}_plot.png")
        img = Image.open(path).convert('RGB')
        img = trim(img, padding=40)
        images.append(img)
    cols = 3
    rows = 2
    
    max_w = max(img.size[0] for img in images)
    max_h = max(img.size[1] for img in images)
    
    row_gap = 80 # Reduced vertical buffer
    col_gap = 50 # Standard horizontal split
    
    cell_w = max_w + col_gap
    cell_h = max_h + row_gap
    
    grid = Image.new('RGB', (cell_w * cols, cell_h * rows), color=(255,255,255))
    draw = ImageDraw.Draw(grid)
    
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        
        x_offset = c * cell_w + (cell_w - img.size[0]) // 2
        y_offset = r * cell_h + (row_gap // 2)
        
        grid.paste(img, (x_offset, y_offset))
        
        # Mark (a) to the top left of the centered figure. 
        # y_offset is visually aligned with the original title top boundary.
        draw.text((x_offset + 30, y_offset + 10), letters[i], fill=(0,0,0), font=font)
        
    out_path = f"paper/{prefix}_combined.pdf"
    grid.save(out_path, "PDF", resolution=300.0)
    print(f"Saved {out_path}")

stitch('pca')
stitch('tsne')
