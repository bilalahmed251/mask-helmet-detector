import os
import xml.etree.ElementTree as ET
from PIL import Image
import random
import shutil

# Paths
ANNOTATIONS_DIR = "annotations"
IMAGES_DIR = "images"
DATASET_DIR = "dataset/mask"

TRAIN_SPLIT = 0.8

def prepare():
    if os.path.exists(DATASET_DIR):
        print(f"Cleaning up old dataset dir: {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)
        
    for split in ['train', 'val']:
        for label in ['with_mask', 'without_mask']:
            os.makedirs(os.path.join(DATASET_DIR, split, label), exist_ok=True)
            
    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]
    random.seed(42)
    random.shuffle(xml_files)
    
    # Limit to exactly 400 images to speed up training time!
    xml_files = xml_files[:400]
    
    split_idx = int(len(xml_files) * TRAIN_SPLIT)
    train_files = xml_files[:split_idx]
    
    counts = {"with_mask": 0, "without_mask": 0}
    
    print("Processing images and cropping faces...")
    for idx, xml_file in enumerate(xml_files):
        split = 'train' if xml_file in train_files else 'val'
        tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
        root = tree.getroot()
        
        filename = root.find('filename').text
        img_path = os.path.join(IMAGES_DIR, filename)
        
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGES_DIR, xml_file.replace('.xml', '.png'))
        
        if not os.path.exists(img_path):
            print(f"Missing image for {xml_file}")
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue
            
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label == "mask_weared_incorrect":
                label = "without_mask"  # treat incorrect as without mask for safety
                
            if label not in ['with_mask', 'without_mask']:
                continue
                
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Crop and save
            face = img.crop((xmin, ymin, xmax, ymax))
            
            counts[label] += 1
            face_filename = f"{filename.split('.')[0]}_{counts[label]}.jpg"
            face_path = os.path.join(DATASET_DIR, split, label, face_filename)
            face.save(face_path, "JPEG")
            
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(xml_files)} files...")
            
    print(f"Done! Generated dataset structure in {DATASET_DIR}")

if __name__ == "__main__":
    prepare()
