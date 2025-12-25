import gradio as gr
import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import shutil
import base64
from io import BytesIO

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
from PaddleOCR import PaddleOCR, draw_ocr

FONT = './PaddleOCR/doc/fonts/latin.ttf'

# Global models
detector = None
recognitor = None

def load_models():
    global detector, recognitor
    if detector is None:
        detector = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, use_gpu=False)
    if recognitor is None:
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        recognitor = Predictor(config)

def predict_single_image(img_path):
    load_models()

    img = cv2.imread(img_path)
    result = detector.ocr(img_path, cls=False, det=True, rec=False)
    boxes = result[0]

    img_pil = Image.open(img_path)
    texts = []
    for box in boxes:
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x_min, x_max = int(min(x_coords) - 4), int(max(x_coords) + 4)
        y_min, y_max = int(min(y_coords) - 4), int(max(y_coords) + 4)
        cropped_img = img_pil.crop((x_min, y_min, x_max, y_max))
        text = recognitor.predict(cropped_img)
        texts.append(text)

    # Create annotated image
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    extra_width = 600
    new_image = np.zeros((height, width + extra_width, 3), dtype=np.uint8)
    new_image[:, :width] = image

    shifted_boxes = []
    for box in boxes:
        shifted_box = [(point[0] + width, point[1]) for point in box]
        shifted_boxes.append(shifted_box)

    font_path = FONT if os.path.exists(FONT) else None
    annotated = draw_ocr(new_image, shifted_boxes, texts, font_path=font_path)

    return boxes, texts, Image.fromarray(annotated)

def extract_invoice_fields(raw_texts):
    import requests
    import time

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        return {"error": "GOOGLE_API_KEY not set"}

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
    6. Với bản vẽ kỹ thuật: Thông tin như Material, Finish, Projection Method, Size, Scale, v.v. THUỘC VỀ "technical_drawing.metadata", KHÔNG phải "technical_specs". Chỉ dùng "technical_specs" cho các thông số kỹ thuật lẻ không nằm trong metadata (ví dụ: trọng lượng, kích thước ống nếu có bảng riêng).

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
                "source_text": "Văn bản nguồn từ OCR cho tên hàng hóa",
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
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0
        }
    }

    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result:
                    return json.loads(result['candidates'][0]['content']['parts'][0]['text'])
            time.sleep(2)
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(2)

    return {"error": "Failed to extract data"}

def display_input_image(images):
    if images:
        # Get the first image path
        if isinstance(images[0], tuple):
            img_path = images[0][0]  # filepath from tuple
        else:
            img_path = images[0]  # direct path
        return Image.open(img_path)
    return None

def process_images(images):
    if not images:
        return None, None, "Vui lòng upload ít nhất một ảnh."

    with tempfile.TemporaryDirectory() as temp_dir:
        all_texts = []
        annotated_images = []
        input_images = []

        for img_file in images:
            # Gradio returns tuple (filepath, filename) for file inputs
            if isinstance(img_file, tuple):
                img_path = img_file[0]  # filepath
            else:
                img_path = img_file  # fallback

            # Load input image
            input_img = Image.open(img_path)
            input_images.append(input_img)

            # Process
            boxes, texts, annotated_img = predict_single_image(img_path)
            annotated_images.append(annotated_img)

            # Sort texts by position
            sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0][1])
            sorted_texts = [texts[i] for i in sorted_indices]
            all_texts.extend(sorted_texts)

        # Extract structured data
        ai_result = extract_invoice_fields(all_texts)
        final_data = ai_result.copy()
        final_data["raw_text_all"] = all_texts

        # Save JSON
        json_path = os.path.join(temp_dir, "result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        # Generate HTML table for items if present
        table_html = ""
        if final_data.get("items") and isinstance(final_data["items"], list) and len(final_data["items"]) > 0:
            table_html = """
            <h3>Pricing Table</h3>
            <table style="width:100%; border-collapse:collapse; font-family:Arial, sans-serif;">
                <thead>
                    <tr style="background-color:#f2f2f2;">
                        <th style="border:1px solid #ddd; padding:8px; text-align:left;">STT</th>
                        <th style="border:1px solid #ddd; padding:8px; text-align:left;">Tên hàng hóa</th>
                        <th style="border:1px solid #ddd; padding:8px; text-align:right;">Số lượng</th>
                        <th style="border:1px solid #ddd; padding:8px; text-align:left;">Đơn vị</th>
                        <th style="border:1px solid #ddd; padding:8px; text-align:right;">Đơn giá</th>
                        <th style="border:1px solid #ddd; padding:8px; text-align:right;">Thành tiền</th>
                        <th style="border:1px solid #ddd; padding:8px; text-align:left;">Ghi chú kỹ thuật</th>
                    </tr>
                </thead>
                <tbody>
            """
            for i, item in enumerate(final_data["items"], 1):
                table_html += f"""
                    <tr>
                        <td style="border:1px solid #ddd; padding:8px;">{i}</td>
                        <td style="border:1px solid #ddd; padding:8px;">{item.get('source_text', item.get('name', ''))}</td>
                        <td style="border:1px solid #ddd; padding:8px; text-align:right;">{item.get('quantity', '')}</td>
                        <td style="border:1px solid #ddd; padding:8px;">{item.get('unit', '')}</td>
                        <td style="border:1px solid #ddd; padding:8px; text-align:right;">{item.get('unit_price', ''):,}</td>
                        <td style="border:1px solid #ddd; padding:8px; text-align:right;">{item.get('total_money', ''):,}</td>
                        <td style="border:1px solid #ddd; padding:8px;">{item.get('specs_ref', '')}</td>
                    </tr>
                """
            table_html += "</tbody></table>"

        # Return annotated image, table HTML, and JSON
        if annotated_images and input_images:
            # Convert PIL Image to base64 for annotated
            img = annotated_images[0]
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_html = f'<img src="data:image/png;base64,{img_str}" style="max-width:100%; height:auto;">'
            return img_html, table_html, json.dumps(final_data, ensure_ascii=False, indent=4)
        else:
            return None, table_html, json.dumps(final_data, ensure_ascii=False, indent=4)

# Gradio Interface
with gr.Blocks(title="Vietnamese OCR with Structured Extraction") as demo:
    gr.Markdown("# Vietnamese OCR with Structured Data Extraction")
    gr.Markdown("Upload ảnh tài liệu (hóa đơn, bản vẽ kỹ thuật, v.v.) để trích xuất văn bản và dữ liệu có cấu trúc.")

    with gr.Row():
        image_input = gr.File(label="Upload Images", file_types=["image"], file_count="multiple")
        process_btn = gr.Button("Process Images")

    with gr.Row():
        input_image_output = gr.Image(label="Input Image")

    with gr.Row():
        annotated_output = gr.HTML(label="Annotated Image")

    with gr.Row():
        table_output = gr.HTML(label="Pricing Table")

    with gr.Row():
        json_output = gr.JSON(label="Extracted Data")

    image_input.change(display_input_image, inputs=image_input, outputs=input_image_output)
    process_btn.click(process_images, inputs=image_input, outputs=[annotated_output, table_output, json_output])

if __name__ == "__main__":
    demo.launch()
