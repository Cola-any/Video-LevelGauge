import os
import json
from tqdm import tqdm
import os
import cv2

def extract_6_frames_uniformly(video_paths, save_root):
    os.makedirs(save_root, exist_ok=True)
    for video_path in tqdm(video_paths, desc="Extracting 6 frames per video"):
        video_name = os.path.basename(video_path).replace(".mp4", "")
        save_dir = os.path.join(save_root, video_name)
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 6:
            print(f"Warning: {video_path} has less than 6 frames, skipping.")
            cap.release()
            continue
        indices = [int(i * (total_frames - 1) / 5) for i in range(6)]
        extracted = 0
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id in indices:
                resized_frame = cv2.resize(frame, (800,600))
                frame_filename = os.path.join(save_dir, f"frame{extracted}.jpg")
                cv2.imwrite(frame_filename, resized_frame)
                extracted += 1
                if extracted == 6:
                    break
            frame_id += 1
        cap.release()
    print("Done: All videos processed.")


if __name__ == "__main__":
    json_file = './LevelGauge/json/Pos_MCQA_300_final.json'
    with open(json_file, "r") as f:
            data = json.load(f)
    all_video_names = set()
    for item in data:
        if "video_name" in item:
            path = os.path.join('./LevelGauge/videos', item["video_name"])
            all_video_names.add(path)
        if "9_background" in item:
            for bg_name in item["9_background"].keys():
                path = os.path.join('./LevelGauge/videos', bg_name)
                all_video_names.add(path)
    # all_video_names = sorted(all_video_names)
    print(len(all_video_names))
    extract_6_frames_uniformly(
        video_paths=all_video_names,
        save_root='./LevelGauge/frame'
    )