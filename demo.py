import os
import sys
import argparse
import time
import threading
import cv2
import torch
import numpy as np

# Ensure Modules directory is in path
sys.path.insert(0, os.path.abspath("."))

from Modules.Model import yolo26n_custom

# --- 80 COCO Class Names ---
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# --- Threaded Video Stream for High FPS ---
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
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

# --- Classification Heatmap Visualization ---
def get_heatmaps(model, outputs, class_idx, mode="one2one"):
    """
    Extracts the classification heatmaps for a given class index.
    
    Args:
        model: YOLO26_Custom model
        outputs: The full outputs from the model forward pass (y, preds)
        class_idx (int): The index of the class to visualize
        mode (str): Branch to extract heatmap from ("one2one" or "one2many")
        
    Returns:
        List of 3 numpy arrays (2D) containing the heatmaps at 3 scales.
    """
    if not isinstance(outputs, tuple) or len(outputs) < 2:
        return None
        
    preds = outputs[1]
    
    # Check if end2end (one-to-one) is used and select appropriate branch
    if isinstance(preds, dict) and "one2one" in preds:
        if mode == "one2many" and "one2many" in preds:
            feats = preds["one2many"]["feats"]
            cls_head = model.head.cv3
        else:  # default to one2one
            feats = preds["one2one"]["feats"]
            cls_head = model.head.one2one_cv3
    elif isinstance(preds, dict) and "feats" in preds:
        feats = preds["feats"]
        cls_head = model.head.cv3
    else:
        return None
        
    heatmaps = []
    for i in range(len(feats)):
        # Forward through the classification head at scale i
        # feats[i] has shape (1, C, H, W)
        cls_out = cls_head[i](feats[i])  # Shape: (1, nc, H, W)
        # Apply sigmoid to get confidence scores [0, 1]
        scores = torch.sigmoid(cls_out)
        # Extract the heatmap for the specified class index
        heatmap = scores[0, class_idx].detach().cpu().numpy()  # Shape: (H, W)
        heatmaps.append(heatmap)
        
    return heatmaps

def show_heatmap_window(frame, heatmaps, active_class_idx, class_names, mode="one2one"):
    if heatmaps is None:
        return
        
    h_orig, w_orig = frame.shape[:2]
    # Resize each overlay to 320px width (maintaining aspect ratio)
    target_w = 320
    target_h = int(h_orig * (target_w / w_orig))
    
    strides = [8, 16, 32]
    heatmap_overlays = []
    
    for i, hm in enumerate(heatmaps):
        h_i, w_i = hm.shape
        max_val = hm.max()
        total = (hm > 0.25).sum().item()
        # Scale color to absolute range [0, 1] where closer to 1 is more intense
        hm_norm = np.clip(hm, 0.0, 1.0)
            
        hm_uint8 = (hm_norm * 255).astype(np.uint8)
        hm_resized = cv2.resize(hm_uint8, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        hm_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
        
        # Overlay on clean frame copy
        overlay = cv2.addWeighted(frame, 0.5, hm_color, 0.5, 0)
        overlay_small = cv2.resize(overlay, (target_w, target_h))
        
        # Add stride text label
        label_text = f"Stride {strides[i]} ({h_i}x{w_i}) Max: {max_val:.2f} Total: {total}"
        cv2.putText(
            overlay_small,
            label_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )
        heatmap_overlays.append(overlay_small)
        
    # Concatenate horizontally
    combined_heatmap = np.hstack(heatmap_overlays)
    
    # Add header
    class_name = class_names[active_class_idx] if active_class_idx < len(class_names) else f"class_{active_class_idx}"
    header = np.zeros((40, combined_heatmap.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"Heatmaps ({mode}) for Class: {class_name} (ID: {active_class_idx})",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA
    )
    
    final_heatmap_window = np.vstack([header, combined_heatmap])
    cv2.imshow("Classification Heatmaps", final_heatmap_window)

def main():
    parser = argparse.ArgumentParser(description="YOLO26 Custom Demo")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file (.pt)")
    parser.add_argument("--standard", type=str, default=None, help="Standard Ultralytics model name (e.g., yolo26m.pt)")
    parser.add_argument("--nc", type=int, default=80, help="Number of classes (80 for COCO, 1 for Face/LFW)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--image", type=str, default=None, help="Path to input image for single-image demo")
    parser.add_argument("--heatmap", action="store_true", help="Show heatmap of the classification head")
    parser.add_argument("--heatmap-class", type=str, default="auto", help="Class name or index to show heatmap for (default: auto)")
    parser.add_argument("--heatmap-mode", type=str, default="one2one", choices=["one2one", "one2many"], help="Detection branch for heatmap (one2one or one2many)")
    args = parser.parse_args()

    # Check for mutual exclusivity of --weights and --standard
    if args.weights is not None and args.standard is not None:
        parser.error("Cannot use both --weights and --standard flags at the same time.")

    # If neither is specified, default to using yolo26n_custom_distilled.pt weights
    if args.weights is None and args.standard is None:
        args.weights = "yolo26n_custom_distilled.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Load Model and Weights ---
    if args.standard:
        print(f"Loading standard Ultralytics model: {args.standard}...")
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(args.standard)
            model = yolo_model.model.to(device)
            # Aliasing model.head to model.model[-1] for compatibility with heatmap extraction
            if hasattr(model, "model") and len(model.model) > 0:
                model.head = model.model[-1]
            print("Standard model loaded successfully!")
            
            # Map names from ultralytics model
            if hasattr(yolo_model, "names") and yolo_model.names:
                class_names = [yolo_model.names[i] for i in range(len(yolo_model.names))]
                args.nc = len(class_names)
            elif hasattr(model, "nc"):
                args.nc = model.nc
                class_names = [f"class_{i}" for i in range(args.nc)]
            else:
                class_names = COCO_CLASSES if args.nc == 80 else [f"class_{i}" for i in range(args.nc)]
        except Exception as e:
            print(f"Error loading standard model: {e}")
            sys.exit(1)
    else:
        # Determine class names mapping
        if args.nc == 80:
            class_names = COCO_CLASSES
        elif args.nc == 1:
            class_names = ["face"]  # Default to face for LFW or single class face detector
        else:
            class_names = [f"class_{i}" for i in range(args.nc)]

        print(f"Loading custom model with {args.nc} classes...")
        model = yolo26n_custom(nc=args.nc, end2end=True).to(device)
        
        if os.path.exists(args.weights):
            print(f"Loading weights from {args.weights}...")
            try:
                state_dict = torch.load(args.weights, map_location=device)
                model.load_state_dict(state_dict)
                print("Weights loaded successfully!")
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("Running with randomly initialized weights.")
        else:
            print(f"Warning: Weights file not found at '{args.weights}'. Running with randomly initialized weights.")

    # Determine class index for heatmap
    heatmap_class_idx = 0
    if args.heatmap:
        if args.heatmap_class.lower() == "auto":
            heatmap_class_idx = "auto"
        else:
            try:
                heatmap_class_idx = int(args.heatmap_class)
                if heatmap_class_idx < 0 or heatmap_class_idx >= args.nc:
                    print(f"Warning: Heatmap class index {heatmap_class_idx} out of range (0-{args.nc-1}). Defaulting to 0.")
                    heatmap_class_idx = 0
            except ValueError:
                try:
                    heatmap_class_idx = class_names.index(args.heatmap_class.lower())
                except ValueError:
                    print(f"Warning: Class '{args.heatmap_class}' not found in class list. Defaulting to 0.")
                    heatmap_class_idx = 0

    model.eval()

    # --- 2. Run Single-Image Demo or Webcam Stream ---
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found.")
            sys.exit(1)
        
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Failed to read image from '{args.image}'.")
            sys.exit(1)
            
        h_orig, w_orig = frame.shape[:2]
        
        with torch.no_grad():
            # --- 3. Preprocess Frame for YOLO ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_frame, (640, 640))
            img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 640, 640]

            # --- 4. Forward Pass ---
            outputs = model(img_tensor)
            if isinstance(outputs, tuple):
                y = outputs[0]
            else:
                y = outputs

            detections = y[0].cpu().numpy()  # Shape: [max_det, 6]

            if args.heatmap:
                clean_frame = frame.copy()

            # --- 5. Draw Detections ---
            for det in detections:
                x1, y1, x2, y2, score, class_idx = det
                if score < args.conf:
                    continue

                class_idx = int(class_idx)
                label_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"

                rx1 = int(x1 * (w_orig / 640.0))
                ry1 = int(y1 * (h_orig / 640.0))
                rx2 = int(x2 * (w_orig / 640.0))
                ry2 = int(y2 * (h_orig / 640.0))

                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (46, 204, 113), 2)
                label_str = f"{label_name}: {score:.2f}"
                cv2.putText(
                    frame,
                    label_str,
                    (rx1, max(ry1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (46, 204, 113),
                    2,
                    lineType=cv2.LINE_AA
                )

            if args.heatmap:
                # Determine class index dynamically if 'auto'
                if heatmap_class_idx == "auto":
                    active_class_idx = 0
                    best_score = -1.0
                    for det in detections:
                        _, _, _, _, score, class_idx = det
                        if score >= args.conf and score > best_score:
                            best_score = score
                            active_class_idx = int(class_idx)
                else:
                    active_class_idx = heatmap_class_idx

                heatmaps = get_heatmaps(model, outputs, active_class_idx, mode=args.heatmap_mode)
                show_heatmap_window(clean_frame, heatmaps, active_class_idx, class_names, mode=args.heatmap_mode)
        
        cv2.imshow("YOLO26 Custom Model - Image Demo", frame)
        print("Image demo is running. Press any key on the output window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Demo closed.")
    else:
        print("Starting webcam stream...")
        vs = WebCamStream(src=args.camera, width=640, height=480).start()
        time.sleep(1.0) # Wait for camera sensor to warm up

        print("Webcam demo is running. Press 'q' on the output window to exit.")
        prev_time = time.time()

        with torch.no_grad():
            while True:
                grabbed, frame = vs.read()
                if not grabbed or frame is None:
                    print("Failed to grab frame from camera. Exiting...")
                    break

                h_orig, w_orig = frame.shape[:2]

                # --- 3. Preprocess Frame for YOLO ---
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_frame, (640, 640))
                img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 640, 640]

                # --- 4. Forward Pass ---
                outputs = model(img_tensor)
                if isinstance(outputs, tuple):
                    y = outputs[0]
                else:
                    y = outputs

                detections = y[0].cpu().numpy()  # Shape: [max_det, 6]

                if args.heatmap:
                    clean_frame = frame.copy()

                # --- 5. Draw Detections ---
                for det in detections:
                    x1, y1, x2, y2, score, class_idx = det
                    if score < args.conf:
                        continue

                    class_idx = int(class_idx)
                    label_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"

                    rx1 = int(x1 * (w_orig / 640.0))
                    ry1 = int(y1 * (h_orig / 640.0))
                    rx2 = int(x2 * (w_orig / 640.0))
                    ry2 = int(y2 * (h_orig / 640.0))

                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (46, 204, 113), 2)
                    label_str = f"{label_name}: {score:.2f}"
                    cv2.putText(
                        frame,
                        label_str,
                        (rx1, max(ry1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (46, 204, 113),
                        2,
                        lineType=cv2.LINE_AA
                    )

                if args.heatmap:
                    # Determine class index dynamically if 'auto'
                    if heatmap_class_idx == "auto":
                        active_class_idx = 0
                        best_score = -1.0
                        for det in detections:
                            _, _, _, _, score, class_idx = det
                            if score >= args.conf and score > best_score:
                                best_score = score
                                active_class_idx = int(class_idx)
                    else:
                        active_class_idx = heatmap_class_idx

                    heatmaps = get_heatmaps(model, outputs, active_class_idx, mode=args.heatmap_mode)
                    show_heatmap_window(clean_frame, heatmaps, active_class_idx, class_names, mode=args.heatmap_mode)

                # --- 6. Calculate and Display FPS ---
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
                prev_time = curr_time

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

                # Display output window
                cv2.imshow("YOLO26 Custom Model Demo", frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Clean up
        vs.stop()
        cv2.destroyAllWindows()
        print("Demo closed.")

if __name__ == "__main__":
    main()
