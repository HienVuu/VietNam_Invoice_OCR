import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import argparse
import json
import numpy as np
import openai
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
    Bạn là chuyên gia xử lý dữ liệu hóa đơn (Invoice/Receipt).
    Dưới đây là văn bản OCR thô từ một hóa đơn:
    --- BẮT ĐẦU ---
    {full_text}
    --- KẾT THÚC ---

    NHIỆM VỤ: Trích xuất tối đa thông tin có thể tìm thấy và điền vào JSON dưới đây.
    QUY TẮC:
    - Trường nào không tìm thấy thông tin trong ảnh -> Trả về null (không được bịa ra).
    - Các con số (tiền, lượng) -> Để dạng số (number), bỏ dấu phẩy ngăn cách nghìn.
    - Ngày tháng -> Chuẩn hóa thành DD/MM/YYYY nếu có thể.

    CẤU TRÚC JSON YÊU CẦU:
    {{
        "document_type": "Loại tài liệu (invoice, receipt, prescription, clothing_receipt, fuel_receipt, etc.)",
        "seller": {{
            "name": "Tên đơn vị bán hàng",
            "tax_id": "Mã số thuế bên bán (MST)",
            "address": "Địa chỉ bên bán",
            "phone": "Số điện thoại",
            "bank_account": "Số tài khoản ngân hàng (nếu có)"
        }},
        "buyer": {{
            "name": "Tên người mua / Đơn vị mua",
            "tax_id": "Mã số thuế bên mua",
            "address": "Địa chỉ bên mua"
        }},
        "invoice_info": {{
            "code": "Số hóa đơn (Invoice No)",
            "symbol": "Ký hiệu/Mẫu số (Serial No)",
            "date": "Ngày lập hóa đơn",
            "time": "Giờ in hóa đơn (nếu có)",
            "cashier": "Tên thu ngân (Cashier)",
            "table_no": "Số bàn (dùng cho nhà hàng)"
        }},
        "items": [
            {{
                "sku": "Mã hàng hóa (nếu có)",
                "name": "Tên hàng hóa/dịch vụ",
                "unit": "Đơn vị tính (Cái, Lon, Kg...)",
                "quantity": 1,
                "unit_price": 0,
                "discount": 0,
                "tax_rate": "Thuế suất từng món (5%, 8%, 10%... nếu có)",
                "total_money": 0
            }}
        ],
        "financials": {{
            "subtotal": "Tổng tiền hàng (trước thuế)",
            "total_vat_amount": "Tổng tiền thuế VAT",
            "total_discount": "Tổng tiền chiết khấu",
            "shipping_fee": "Phí vận chuyển (nếu có)",
            "total_payment": "Tổng cộng thanh toán (Final Amount)",
            "amount_in_words": "Số tiền viết bằng chữ"
        }},
        "prescription_info": {{
            "patient_name": "Tên bệnh nhân",
            "doctor_name": "Tên bác sĩ",
            "medications": [
                {{
                    "name": "Tên thuốc",
                    "dosage": "Liều lượng",
                    "quantity": 0
                }}
            ]
        }},
        "clothing_info": {{
            "brand": "Thương hiệu",
            "size": "Kích cỡ",
            "color": "Màu sắc",
            "material": "Chất liệu"
        }},
        "fuel_info": {{
            "fuel_type": "Loại nhiên liệu",
            "liters": 0,
            "price_per_liter": 0
        }}
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
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result:
                    return json.loads(result['candidates'][0]['content']['parts'][0]['text'])
        except Exception:
            time.sleep(2)

    return {}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, type=str, help='Path to input image')
    parser.add_argument('--output_path', default='./output', type=str, help='Path to output folder')
    args = parser.parse_args()

    print("--- Loading OCR models... ---")

    detector = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, use_gpu=False)
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    recognitor = Predictor(config)

    print("--- Models loaded. Starting prediction... ---")

    boxes, texts = predict(
        recognitor=recognitor,
        detector=detector,
        img_path=args.image_path,
        save_path=args.output_path
    )

    sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0][1])
    sorted_texts = [texts[i] for i in sorted_indices]

    print("--- Sending ALL text to Gemini for full classification... ---")

    ai_result = extract_invoice_fields(sorted_texts)

    final_data = ai_result.copy()
    final_data["raw_text_all"] = sorted_texts

    json_output_path = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.image_path))[0] + '.json')

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"JSON saved to: {json_output_path}")
