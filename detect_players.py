from ultralytics import YOLO
import cv2
import os
import argparse

def detect_and_crop(video_path, output_folder, model_path='best.pt'):
    model = YOLO(model_path)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save first frame for inspection
        if frame_idx == 0:
            cv2.imwrite("first_frame.jpg", frame)
            print("Saved first_frame.jpg for manual inspection")

        results = model(frame)[0]
        print(f"Processing frame {frame_idx}...")
        print(f"  Total detections: {len(results.boxes)}")
        
        player_id = 0
        for box in results.boxes:
            cls = int(box.cls[0])
            print(f"    Detected class ID: {cls}")

            if cls == 2:  # 2 = player
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                player_crop = frame[y1:y2, x1:x2]
                if player_crop.size == 0:
                    print("    Skipping empty crop.")
                    continue

                crop_path = os.path.join(output_folder, f"{frame_idx}_{player_id}.jpg")
                cv2.imwrite(crop_path, player_crop)
                print(f"    Saved player crop: {crop_path}")
                player_id += 1

        frame_idx += 1

    cap.release()
    print("✅ Detection completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Directory to save cropped player images')
    parser.add_argument('--model', default='best.pt', help='YOLO model path (default: best.pt)')
    args = parser.parse_args()

    detect_and_crop(args.video, args.output, args.model)
