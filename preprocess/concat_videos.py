import ffmpeg
import os
import subprocess
import tempfile
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def merge_videos_with_reference(video_paths, ref_index, output_path="output.mp4"):
    probe = ffmpeg.probe(video_paths[ref_index])
    video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
    ref_w = int(video_streams[0]["width"])
    ref_h = int(video_streams[0]["height"])
    ref_fps = eval(video_streams[0]["r_frame_rate"])  # fps

    tmpdir = tempfile.mkdtemp()
    resized_files = []
    for i, path in enumerate(video_paths):
        resized_path = os.path.join(tmpdir, f"resized_{i}.mp4")
        (
            ffmpeg
            .input(path)
            .filter("scale", ref_w, ref_h)  
            .filter("fps", fps=ref_fps)      
            .output(
                resized_path,
                vcodec="libx264", preset="fast", crf=23,  
                pix_fmt="yuv420p",                       
                acodec="aac", audio_bitrate="128k"       
            )
            .overwrite_output()
            .run(quiet=True)
        )
        resized_files.append(resized_path)

    list_path = os.path.join(tmpdir, "file_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for file in resized_files:
            f.write(f"file '{file}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Done: {output_path}")

def process_sample(sample):
    pos = "10-00"
    video_dir = "./LevelGauge/videos"
    background9 = sample["9_background"]
    question_id = sample["question_id"]
    video_name = sample["video_name"]
    video_path = os.path.join(video_dir, video_name)

    pos_num = int(pos[-2:])
    background_list = list(background9.keys())
    background_list_ = [os.path.join(video_dir, s) for s in background_list]
    background_list_.insert(pos_num, video_path)

    output_video_dir = f"./LevelGauge/concated_videos/{pos}"
    os.makedirs(output_video_dir, exist_ok=True)
    output_path = f"{output_video_dir}/{question_id}.mp4"
    if os.path.exists(output_path):
        print(output_path)
        pass
    else:
        merge_videos_with_reference(background_list_, ref_index=pos_num, output_path=output_path)
    return question_id

if __name__ == "__main__":
    gt_file = "./LevelGauge/json/Pos_MCQA_300_final.json"
    gt_qa_pairs = json.load(open(gt_file, "r"))

    num_processes = max(1, cpu_count() - 1)
    with Pool(num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_sample, gt_qa_pairs), total=len(gt_qa_pairs)))
