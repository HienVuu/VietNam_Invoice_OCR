import os
import shutil
from PIL import Image
import argparse

def is_easy_image(img_path):
    """
    Heuristic to classify as easy: high resolution, no blur, no skew.
    This is a simple check; you may need to manually review.
    """
    try:
        img = Image.open(img_path)
        width, height = img.size
        # Easy if resolution > 1000x1000 and not too skewed (assume square-ish)
        if width > 1000 and height > 1000 and abs(width - height) < 500:
            return True
        return False
    except:
        return False

def categorize_images(input_dir, easy_dir, hard_dir):
    os.makedirs(easy_dir, exist_ok=True)
    os.makedirs(hard_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            if is_easy_image(img_path):
                shutil.copy(img_path, os.path.join(easy_dir, filename))
                print(f"Copied {filename} to easy")
            else:
                shutil.copy(img_path, os.path.join(hard_dir, filename))
                print(f"Copied {filename} to hard")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='mc_ocr_dataset/images', help='Directory with images')
    parser.add_argument('--easy_dir', default='dataset/easy/images', help='Output dir for easy images')
    parser.add_argument('--hard_dir', default='dataset/hard/images', help='Output dir for hard images')
    args = parser.parse_args()

    categorize_images(args.input_dir, args.easy_dir, args.hard_dir)
    print("Categorization done. Manually review and adjust if needed.")
