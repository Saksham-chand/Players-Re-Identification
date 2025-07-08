# 🏃‍♂️ Player Re-Identification – Cross-Camera Mapping

## 🎯 Objective

    Map and track players consistently between two different camera angles:  
    - `broadcast.mp4`  
    - `tacticam.mp4`  
    Ensure each player retains the same ID across both views using object detection and visual similarity.

## 📁 Folder Structure

    Player Identification
    ├──cross_camera_reid/
        ├── broadcast_crops/ # auto-generated
        ├── tacticam_crops/ # auto-generated
        ├── videos/
        │ ├── broadcast.mp4
        │ └── tacticam.mp4
        ├── best.pt
        ├── detect_players.py
        ├── match_players.py
    ├── README.md
    └── report.pdf (or report.md)

## ⚙️ Setup Instructions

### 1. Create virtual environment

    python3 -m venv myenv
    source myenv/bin/activate

### 2.  Install dependencies

    pip install -r requirements.txt

## ▶️ Run Instructions

### 1.Run detection and crop players from both videos

    python detect_players.py --video videos/broadcast.mp4 --output broadcast_crops --model best.pt
    python detect_players.py --video videos/tacticam.mp4 --output tacticam_crops --model best.pt

### 2. Match players across views using visual similarity

    python match_players.py --broadcast broadcast_crops --tacticam tacticam_crops

## 📝 Notes
    The best.pt YOLOv8 model detects: ['ball', 'goalkeeper', 'player', 'referee'].
    Only class 2 (player) is used.
    ResNet18 is used for embedding extraction.
    Cosine similarity is used for matching.