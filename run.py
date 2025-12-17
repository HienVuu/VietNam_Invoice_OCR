import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import argparse
import json
import numpy as np
#import openai
import requests
import time

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
from PaddleOCR import PaddleOCR, draw_ocr

FONT = './PaddleOCR/doc/fonts/latin.ttf'

def predict(recognitor, detector, img_path, save_path, structured_data=None, padding=4, dpi=100):
    img = cv2.imread(img_path)

    result = detector.ocr(img_path, cls=False, det=True, rec=False)
    boxes = result[0]

    img = Image.open(img_path)
    texts = []
    for box in boxes:
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x_min, x_max = int(min(x_coords) - padding), int(max(x_coords) + padding)
        y_min, y_max = int(min(y_coords) - padding), int(max(y_coords) + padding)
        cropped_img = img.crop((x_min, y_min, x_max, y_max))
        text = recognitor.predict(cropped_img)
        texts.append(text)

    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    extra_width = 600
    new_image = np.zeros((height, width + extra_width, 3), dtype=np.uint8)
    new_image[:, :width] = image

    font_path = os.path.join('.', FONT)

    if structured_data:
        section_colors = {
            "header": (255, 0, 0),
            "body": (0, 255, 0),
            "footer": (0, 0, 255)
        }

        annotated_img = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(annotated_img)

        try:
            font = ImageFont.truetype(font_path, 20)
        except:
            font = ImageFont.load_default()

        for box, text in zip(boxes, texts):
            x_coords = [point[0] + width for point in box]
            y_coords = [point[1] for point in box]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            section = None
            for sec, items in structured_data.items():
                if text in items:
                    section = sec
                    break

            color = section_colors.get(section, (0, 0, 0))

            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

            draw.text((x_min, y_min), text, fill=color, font=font)

        label_font = ImageFont.truetype(font_path, 30) if font_path else ImageFont.load_default()
        draw.text((10, 10), "Header (Blue)", fill=(255, 0, 0), font=label_font)
        draw.text((10, 50), "Body (Green)", fill=(0, 255, 0), font=label_font)
        draw.text((10, 90), "Footer (Red)", fill=(0, 0, 255), font=label_font)

        annotated = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
    else:
        shifted_boxes = []
        for box in boxes:
            shifted_box = [(point[0] + width, point[1]) for point in box]
            shifted_boxes.append(shifted_box)
        annotated = draw_ocr(
            new_image, shifted_boxes, texts,
            font_path=font_path
        )

    annotated_img = Image.fromarray(annotated)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    output_file = os.path.join(save_path, os.path.basename(img_path))
    annotated_img.save(output_file, dpi=(dpi, dpi))

    print(f"--- Saved result to {output_file} ---")
    return boxes, texts



def parse_table_html_to_json(html_table):
    from bs4 import BeautifulSoup
    import re

    if not html_table:
        return []

    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table')
    if not table:
        return []

    rows = table.find_all('tr')
    if len(rows) < 2:
        return []

    headers = []
    header_row = rows[0]
    header_cells = header_row.find_all(['th', 'td'])
    for cell in header_cells:
        text = cell.get_text(strip=True)
        text = re.sub(r'\s+', ' ', text)
        headers.append(text)

    table_data = []
    for row in rows[1:]:
        cells = row.find_all(['th', 'td'])
        row_dict = {}
        for i, cell in enumerate(cells):
            if i < len(headers):
                text = cell.get_text(strip=True)
                text = re.sub(r'\s+', ' ', text)
                row_dict[headers[i]] = text
        if row_dict:
            table_data.append(row_dict)

    return table_data



def extract_invoice_fields(raw_texts):
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key: return {}

    full_text = "\n".join(raw_texts)

    prompt = f"""
    Bạn là chuyên gia xử lý và trích xuất dữ liệu từ đa dạng tài liệu (Hóa đơn, Biên lai, Bảng thông số kỹ thuật, Catalogue vật liệu, Bản vẽ kỹ thuật).
    Dưới đây là văn bản OCR thô từ tài liệu:
    --- BẮT ĐẦU ---
    {full_text}
    --- KẾT THÚC ---

    NHIỆM VỤ: Phân tích nội dung và trích xuất thông tin vào JSON theo cấu trúc dưới đây. Nếu là bản vẽ kỹ thuật, trích xuất thông tin metadata, features, tables như mô tả.

    QUY TẮC QUAN TRỌNG:
    1. Xử lý số liệu:
       - Với Hóa đơn (tiền): Để dạng số, bỏ dấu phân cách nghìn.
       - Với Kỹ thuật (kích thước, trọng lượng): GIỮ NGUYÊN cả số và đơn vị (ví dụ: "10.5 mm", "20 inch").
    2. Xử lý Bảng (Table):
       - Nếu gặp bảng thông số kỹ thuật hoặc bảng revision (nhiều hàng/cột), hãy trích xuất vào trường "technical_tables" hoặc "revision_table".
       - Giữ nguyên cấu trúc hàng/cột của bảng.
    3. Chuẩn hóa:
       - Đơn vị (unit): Xác định từ ghi chú như "DIMENSIONS ARE IN MILLIMETERS" → "mm".
       - Ngày: Chuẩn ISO (YYYY-MM-DD), chuyển từ MM/DD/YYYY hoặc D/M/YYYY.
       - Số: float hoặc int.
    4. Provenance: Cho mỗi trường trích xuất, bao gồm source_text từ OCR tokens.
    5. Nếu trường nào không có thông tin -> Trả về null.

    CẤU TRÚC JSON YÊU CẦU:
    {{
        "document_type": "Phân loại tài liệu: invoice, receipt, technical_spec, datasheet, packing_list, technical_drawing, etc.",
        
        // --- PHẦN DÀNH CHO HÓA ĐƠN / MUA BÁN ---
        "general_info": {{
            "code": "Số hiệu tài liệu (Invoice No / Spec No)",
            "date": "Ngày tháng (DD/MM/YYYY)",
            "seller_or_manufacturer": "Tên nhà cung cấp hoặc Nhà sản xuất",
            "buyer_or_project": "Tên người mua hoặc Tên dự án"
        }},
        
        // --- PHẦN DÀNH CHO THÔNG SỐ KỸ THUẬT ---
        "technical_specs": [
            // Trích xuất các thông số lẻ (không nằm trong bảng)
            {{
                "property": "Tên thông số (VD: Material, Finish, Projection Method)",
                "value": "Giá trị kèm đơn vị (VD: Alum 6061-T6, Anodize (Black) 5~10(µm), Third Angle Projection)",
                "source_text": "Văn bản nguồn từ OCR (VD: MATERIAL Alum 6061-T6)"
            }}
        ],
        "technical_tables": [
            // Dùng cho các bảng kích thước, bảng tra trọng lượng, v.v.
            {{
                "table_name": "Tên bảng (nếu có, VD: Bảng kích thước ống)",
                "headers": ["Tên cột 1", "Tên cột 2", "Tên cột 3"],
                "rows": [
                    ["Giá trị hàng 1 cột 1", "Giá trị hàng 1 cột 2", "Giá trị hàng 1 cột 3"],
                    ["Giá trị hàng 2 cột 1", "Giá trị hàng 2 cột 2", "Giá trị hàng 2 cột 3"]
                ]
            }}
        ],

        // --- PHẦN DÀNH CHO BẢN VẼ KỸ THUẬT ---
        "technical_drawing": {{
            "metadata": {{
                "title": "Tiêu đề bản vẽ (string)",
                "drawing_no": "Số bản vẽ (string)",
                "revision": "Số revision (string)",
                "scale": "Tỷ lệ (string, VD: '1:1')",
                "projection": "Phương pháp chiếu (string, VD: 'THIRD ANGLE PROJECTION')",
                "size": "Kích thước (string, VD: 'C')",
                "format_rev": "Revision định dạng (string, VD: 'A')",
                "ges_pn": "GES PN (string)",
                "material": "Vật liệu (string)",
                "finish": "Hoàn thiện (string + numeric range)",
                "tolerances": {{
                    "hole_diameter": "Dung sai đường kính lỗ (string)",
                    "edge": "Dung sai cạnh (string)",
                    "universal": "Dung sai phổ quát (string)"
                }},
                "dimensions_unit": "Đơn vị kích thước (string, VD: 'mm')",
                "sheet": "Trang (string, VD: '1 of 1')",
                "organization": "Tổ chức (string)",
                "confidentiality": "Bảo mật (string)"
            }},
            "approvals": {{
                "drawn_by": {{
                    "name": "Người vẽ (string)",
                    "date": "Ngày vẽ (ISO YYYY-MM-DD)"
                }},
                "engr": {{
                    "name": "Kỹ sư (string)",
                    "date": "Ngày kỹ sư (ISO YYYY-MM-DD)"
                }},
                "checked_by": {{
                    "name": "Người kiểm tra (string)",
                    "date": "Ngày kiểm tra (ISO YYYY-MM-DD)"
                }}
            }},
            "notes": [
                "Ghi chú 1 (string)",
                "Ghi chú 2 (string)"
            ],
            "drawing_dimensions": {{
                "hole_features": [
                    "Mô tả lỗ 1 (string)",
                    "Mô tả lỗ 2 (string)"
                ],
                "radial_features": [
                    "Mô tả bán kính 1 (string)",
                    "Mô tả bán kính 2 (string)"
                ],
                "angular_features": [
                    "Mô tả góc 1 (string)"
                ],
                "linear_dimensions_sample": [
                    "Kích thước tuyến tính 1 (float)",
                    "Kích thước tuyến tính 2 (float)"
                ]
            }}
        }},

        // --- PHẦN CHI TIẾT MUA HÀNG (GIỮ NGUYÊN TỪ CŨ) ---
        "financials": {{
            "total_payment": "Tổng tiền thanh toán",
            "currency": "Đơn vị tiền tệ (VND, USD...)"
        }},
        "items": [
            {{
                "name": "Tên hàng hóa",
                "quantity": 0,
                "unit": "ĐVT",
                "unit_price": 0,
                "total_money": 0,
                "specs_ref": "Ghi chú kỹ thuật kèm theo món hàng (nếu có)"
            }}
        ]
    }}
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            print(f"API Response Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"API Response: {result}")
                if 'candidates' in result:
                    return json.loads(result['candidates'][0]['content']['parts'][0]['text'])
                else:
                    print("No 'candidates' in response")
            else:
                print(f"API Error: {response.text}")
        except Exception as e:
            print(f"Exception during API call: {e}")
            time.sleep(2)

    return {}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths', required=True, nargs='+', type=str, help='Paths to input images')
    parser.add_argument('--output_path', default='./output', type=str, help='Path to output folder')
    args = parser.parse_args()

    print("--- Loading OCR models... ---")

    detector = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, use_gpu=False)
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    recognitor = Predictor(config)

    print("--- Models loaded. Starting prediction... ---")

    all_texts = []
    for img_path in args.image_paths:
        if not os.path.exists(img_path):
            print(f"Error: Image file '{img_path}' does not exist. Please check the path.")
            continue
        boxes, texts = predict(
            recognitor=recognitor,
            detector=detector,
            img_path=img_path,
            save_path=args.output_path
        )
        sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0][1])
        sorted_texts = [texts[i] for i in sorted_indices]
        all_texts.extend(sorted_texts)

    print("--- Sending ALL combined text to Gemini for full classification... ---")

    ai_result = extract_invoice_fields(all_texts)

    final_data = ai_result.copy()
    final_data["raw_text_all"] = all_texts

    # Use the first image's name for the JSON output
    first_image_name = os.path.splitext(os.path.basename(args.image_paths[0]))[0]
    json_output_path = os.path.join(args.output_path, first_image_name + '.json')

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"JSON saved to: {json_output_path}")
