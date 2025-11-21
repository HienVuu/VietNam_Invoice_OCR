import requests
import os

# Lấy key từ máy
api_key = "AIzaSyDN841Pb_kmciHasU6npFy5hE725_TKq_o"

if not api_key:
    print("❌ Chưa có API Key! Hãy chạy lệnh set key trước.")
    exit()

print(f"🔑 Đang kiểm tra với Key: {api_key[:5]}...{api_key[-5:]}")

# Gọi API để liệt kê danh sách model
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

try:
    response = requests.get(url)
    
    if response.status_code == 200:
        print("\n✅ KẾT NỐI THÀNH CÔNG! Danh sách model khả dụng:")
        models = response.json().get('models', [])
        for m in models:
            # Chỉ in ra model tạo nội dung (generateContent)
            if "generateContent" in m['supportedGenerationMethods']:
                print(f" - {m['name']}")
    else:
        print(f"\n❌ LỖI KẾT NỐI ({response.status_code}):")
        print(response.text)

except Exception as e:
    print(f"\n❌ Lỗi chương trình: {e}")