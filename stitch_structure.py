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

mapping = [
    ('Atlas_h_m6A', '(a) Stem-Loop Spatial Interactors (m6A)'),
    ('Atlas_h_m5C', '(b) Hairpin Configuration (m5C)'),
    ('Atlas_m_m1A', '(c) Complex Watson-Crick Topology (m1A)'),
    ('RMBase_h_PseudoU', '(d) Non-Canonical Bonding Arc Model (PseudoU)')
]

base_dir = 'result_inter/structure'

try:
    font_large = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf', 70)
    font_small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 55)
except:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

def stitch():
    images = []
    labels = []
    
    for ds_name, caption in mapping:
        ds_dir = os.path.join(base_dir, ds_name)
        # Find all pngs and pick the one with the highest index, avoiding contact_map_0
        pngs = [f for f in os.listdir(ds_dir) if f.endswith('.png')]
        best_png = max(pngs, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        full_path = os.path.join(ds_dir, best_png)
        
        print(f"Loading {full_path} for {ds_name}...")
        img = Image.open(full_path).convert('RGB')
        img = trim(img, padding=40)
        images.append(img)
        labels.append(caption)
    
    cols = 2
    rows = 2
    
    max_w = max(img.size[0] for img in images)
    max_h = max(img.size[1] for img in images)
    
    row_gap = 50  # Reduced gap since lower titles are removed
    col_gap = 60 
    
    cell_w = max_w + col_gap
    cell_h = max_h + row_gap
    
    grid = Image.new('RGB', (cell_w * cols, cell_h * rows), color=(255,255,255))
    draw = ImageDraw.Draw(grid)
    
    letters = ['(a)', '(b)', '(c)', '(d)']
    
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        
        x_offset = c * cell_w + (cell_w - img.size[0]) // 2
        y_offset = r * cell_h + (row_gap // 2)
        
        grid.paste(img, (x_offset, y_offset))
        
        # Mark purely the character label on the top left analogous to other plots
        draw.text((x_offset + 30, y_offset + 10), letters[i], fill=(0,0,0), font=font_large)
        
    out_path = f"paper/structure_combined.pdf"
    grid.save(out_path, "PDF", resolution=300.0)
    print(f"Saved {out_path}")

stitch()
