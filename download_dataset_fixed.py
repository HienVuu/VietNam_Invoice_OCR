from datasets import load_dataset
import os
import json
import base64
from PIL import Image
import io

# Tải dataset từ Hugging Face, chỉ lấy 100 mẫu đầu
dataset = load_dataset("5CD-AI/Viet-Receipt-VQA", split='train[:100]')

# Tạo thư mục cho dataset
os.makedirs("mc_ocr_dataset", exist_ok=True)
os.makedirs("mc_ocr_dataset/images", exist_ok=True)
os.makedirs("mc_ocr_dataset/annotations", exist_ok=True)

print("Dataset loaded. Check structure:")
print(dataset.column_names)
print(dataset[0])

# Lưu images trước
for i, item in enumerate(dataset):
    img_path = f"mc_ocr_dataset/images/image_{i}.jpg"
    try:
        if 'image' in item:
            if isinstance(item['image'], str):
                # Giả sử base64
                img_data = base64.b64decode(item['image'])
                img = Image.open(io.BytesIO(img_data))
                img.save(img_path)
            elif hasattr(item['image'], 'save'):
                # PIL Image
                item['image'].save(img_path)
                print(f"Saved image {i}")
            else:
                print(f"Unknown image format for {i}")
        else:
            print(f"No image for {i}")
    except Exception as e:
        print(f"Failed to save image {i}: {e}")

# Lưu annotations as list of dicts, bỏ image
ann_list = []
for item in dataset:
    item_dict = dict(item)
    item_dict.pop('image', None)
    ann_list.append(item_dict)

with open("mc_ocr_dataset/annotations/annotations.json", 'w', encoding='utf-8') as f:
    json.dump(ann_list, f, ensure_ascii=False, indent=4)

print("Dataset downloaded to mc_ocr_dataset/ (100 samples)")
