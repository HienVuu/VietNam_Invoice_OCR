# Kế hoạch tạo bộ dữ liệu Ground Truth cho Vietnamese OCR

## Bước 1: Chuẩn bị môi trường
- Đã cài đặt dependencies từ requirement.txt (đang chạy).
- Cần cài đặt PaddleOCR: `pip install paddlepaddle paddleocr` (cho CPU, nếu có GPU thì paddlepaddle-gpu).
- Đảm bảo có Google API Key cho Gemini (đặt biến môi trường GOOGLE_API_KEY).

## Bước 2: Truy cập bộ dữ liệu MC-OCR 2021
- Bộ dữ liệu có 2034 ảnh hóa đơn tiếng Việt với annotations từ Gemini 1.5 Flash.
- Tải về từ nguồn chính thức (có thể là Hugging Face hoặc GitHub của MC-OCR).
- Giả sử bạn đã có, đặt vào thư mục `mc_ocr_dataset/` với cấu trúc:
  - mc_ocr_dataset/images/ (2034 ảnh)
  - mc_ocr_dataset/annotations/ (file JSON hoặc CSV với 14,238 annotations)

## Bước 3: Phân loại ảnh thành Easy và Hard
- Tạo thư mục: `dataset/easy/images/`, `dataset/easy/jsons/`, `dataset/hard/images/`, `dataset/hard/jsons/`.
- Duyệt qua từng ảnh trong mc_ocr_dataset/images/:
  - Nếu ảnh chụp thẳng, rõ ràng, không nhăn -> copy vào dataset/easy/images/
  - Nếu ảnh mờ, lệch, nhăn -> copy vào dataset/hard/images/
- Ước tính: khoảng 50-50% cho easy và hard.

## Bước 4: Tạo Ground Truth JSON
- Cho mỗi ảnh trong dataset/easy/images/ và dataset/hard/images/:
  - Chạy pipeline: `python run.py --image_paths path/to/image.jpg --output_path temp_output`
  - File JSON được tạo sẽ có cấu trúc như output/image.json (seller, buyer, items, etc.).
  - Mở JSON, so sánh với annotations từ MC-OCR (nếu có), và chỉnh sửa bằng tay để đảm bảo chính xác.
  - Lưu JSON ground truth vào thư mục tương ứng (easy/jsons/ hoặc hard/jsons/), với tên giống ảnh (e.g., image.jpg -> image.json).

## Bước 5: Validation và Mở rộng
- Chạy pipeline trên một subset (10-20 ảnh) và so sánh output với ground truth để đo độ chính xác.
- Nếu cần, thêm augmentations cho hard set (blur, skew) để tăng dữ liệu.
- Cuối cùng, có bộ dữ liệu với ~1000+ ảnh mỗi loại, ready để train/eval improvements.

## Công cụ hỗ trợ
- Sử dụng script để batch process: Tạo file process_dataset.py để tự động chạy run.py trên nhiều ảnh.
- Nếu annotations MC-OCR là structured, parse chúng thành JSON schema của project.
