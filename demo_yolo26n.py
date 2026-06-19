import sys
import time
import threading

# 1. Kiểm tra các thư viện cần thiết
try:
    import cv2
except ImportError:
    print("Error: Thư viện 'opencv-python' chưa được cài đặt.")
    print("Vui lòng chạy lệnh sau để cài đặt:")
    print("    pip install opencv-python")
    sys.exit(1)

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("Error: Thư viện 'ultralytics' và 'torch' chưa được cài đặt.")
    print("Vui lòng chạy lệnh sau để cài đặt:")
    print("    pip install ultralytics torch")
    sys.exit(1)


# 2. Lớp đọc Camera bằng luồng (Threaded Video Stream) giúp tối đa hóa FPS
# Luồng này chạy ngầm liên tục cập nhật frame mới nhất từ webcam để main thread không bị block
class WebCamStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.started = False
                break
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            # Sao chép nhanh frame để tránh xung đột ghi/đọc luồng
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()


def main():
    # Tối ưu hóa cài đặt OpenCV
    cv2.setUseOptimized(True)

    # 3. Ép buộc sử dụng CPU theo yêu cầu
    device = "cpu"
    half_precision = False
    print("💻 Chạy ở chế độ: Chỉ sử dụng CPU (CPU-only).")

    # 4. Tải mô hình YOLO26 Nano
    print("Đang tải mô hình YOLO26 Nano...")
    try:
        model = YOLO("yolo26n.pt")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        sys.exit(1)

    # 5. Khởi động luồng đọc camera
    print("Đang mở camera bằng luồng tối ưu...")
    vs = WebCamStream(src=0, width=640, height=480).start()
    time.sleep(1.0)  # Chờ camera ổn định

    print("\n=== ĐANG CHẠY DEMO YOLO26 NANO - PERSON DETECT ===")
    print(f"Thiết bị: {device.upper()} | Độ phân giải YOLO: 320x320 | FP16: {half_precision}")
    print("Nhấn phím 'q' trên cửa sổ hiển thị để THOÁT.")

    prev_time = time.time()
    
    # Disable gradient tính toán để tăng tốc inference
    with torch.no_grad():
        while True:
            grabbed, frame = vs.read()
            if not grabbed or frame is None:
                print("Không nhận được hình ảnh từ camera. Đang thoát...")
                break

            # 6. Chạy YOLO26 Nano với các tham số tối ưu tốc độ tối đa:
            # - imgsz=320: giảm kích thước ảnh đầu vào (mặc định 640) giúp tăng tốc gấp ~4 lần
            # - device=device: chạy trên GPU nếu có
            # - half=half_precision: dùng FP16 để giảm dung lượng tính toán trên GPU
            # - classes=[0]: chỉ nhận diện người (class 0)
            # - verbose=False: tắt các dòng log in ra màn hình terminal
            results = model.predict(
                source=frame,
                imgsz=320,
                device=device,
                half=half_precision,
                classes=[0],
                verbose=False
            )

            # 7. Vẽ Bounding Box người
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    px1, py1, px2, py2 = map(int, xyxy)
                    conf = float(box.conf[0].cpu().numpy())

                    # Vẽ khung hình chữ nhật và nhãn
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (46, 204, 113), 2)
                    cv2.putText(
                        frame,
                        f"Person: {conf:.2f}",
                        (px1, max(py1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (46, 204, 113),
                        2,
                        lineType=cv2.LINE_AA
                    )

            # 8. Tính toán FPS thực tế của vòng lặp chính
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Vẽ chỉ số FPS lên màn hình
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA
            )

            # Hiển thị
            cv2.imshow("YOLO26 Nano Max Speed Demo", frame)

            # Thoát nếu nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Dọn dẹp tài nguyên
    vs.stop()
    cv2.destroyAllWindows()
    print("Đã đóng chương trình.")


if __name__ == "__main__":
    main()
