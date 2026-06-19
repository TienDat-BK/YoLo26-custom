import os
import shutil
import sys

try:
    import ultralytics
except ImportError:
    print("Error: Thư viện 'ultralytics' chưa được cài đặt trong môi trường python này.")
    print("Vui lòng chạy lệnh: pip install ultralytics")
    sys.exit(1)

# Tìm đường dẫn gốc của thư viện ultralytics
ultralytics_dir = os.path.dirname(ultralytics.__file__)

# Đường dẫn đến file yolo26.yaml
yaml_relative_path = os.path.join("cfg", "models", "26", "yolo26.yaml")
source_path = os.path.join(ultralytics_dir, yaml_relative_path)

# Đường dẫn đích (xuất ra ngay thư mục hiện tại)
destination_path = os.path.join(os.getcwd(), "yolo26.yaml")

if os.path.exists(source_path):
    shutil.copy(source_path, destination_path)
    print(f" Thành công! Đã export file cấu hình ra ngoài.")
    print(f"📍 Vị trí file: {destination_path}")
    print("Bạn có thể mở file này bằng VS Code, Notepad hoặc bất kỳ trình soạn thảo nào để đọc.")
else:
    print(f"❌ Không tìm thấy file yolo26.yaml tại đường dẫn mặc định: {source_path}")
    print("Có thể phiên bản ultralytics của bạn cũ hơn phiên bản hỗ trợ YOLO26. Hãy thử cập nhật: pip install -U ultralytics")