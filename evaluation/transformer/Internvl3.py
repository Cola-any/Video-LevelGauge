import argparse
import os
import json
from tqdm import tqdm
import torch
import numpy as np
from decord import VideoReader, cpu
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

NEI_N_FRAMES = 44
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Directory containing model files.", required=False, default='OpenGVLab/InternVL3-8B')
    parser.add_argument("--video_dir", help="Directory containing video files.", required=False, default='./LevelGauge/videos')
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question and answer.", required=False, default='./LevelGauge/json/Pos_MCQA_300_final.json')
    parser.add_argument("--output_dir", help="Directory to save the model response.", required=False, default='./internvl_output')
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=False, default='10-00')
    parser.add_argument("--pos", type=str, help="10-00 represents that probe is inserted in the first position, and 10-10 represents probe input only.", default="10-00")
    parser.add_argument("--num_frames", type=int, default=6)
    return parser.parse_args()

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def get_option_prompt(candidates, version="default"):
    option_prompt = ""
    options = []
    for idx, candidate in enumerate(candidates):
        choice = chr(ord("A") + idx)
        if version == "v4":
            option_prompt += f"({choice}) {candidate}\n"
        else:
            option_prompt += f"({choice}):{candidate} "
        options.append(choice)
    options = "(" + ",".join(options) + ")"
    return option_prompt

def pos_inference(
    video_path,
    background9,
    question,
    candidates,
    model,
    tokenizer,
    frame_number,
    pos,
    video_dir
):
    candidates_prompt = get_option_prompt(candidates, version='v4')
    prompt = f'Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n{candidates_prompt}\n'
    prompt = prompt + "Answer with the option's letter from the given choices directly."
    
    if pos[:2] == "10":
        pos_num = int(pos[-2:])
        #---------------- probe input only
        if pos_num == int(pos[:2]):
            pixel_values, num_patches_lists = load_video(video_path, num_segments=frame_number, max_num=1)
        #---------------- insert probe into the background
        else:
            background_list = list(background9.keys())
            background_list_ = [video_dir + "/" + s for s in background_list]
            background_list_.insert(pos_num, video_path)
            pixel_values_list = []
            num_patches_lists = []
            for video_path_ in background_list_:
                pixel_values, num_patches_list = load_video(video_path_, num_segments=frame_number, max_num=1)
                pixel_values_list.append(pixel_values)
                num_patches_lists = num_patches_lists + num_patches_list
            pixel_values = torch.cat(pixel_values_list, dim=0)
   
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_lists))])
    question = video_prefix + prompt

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response, _ = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_lists, history=None, return_history=True)
    return response

def run_inference(args):
    path = args.model_dir
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    print(args.gt_file)
    
    gt_qa_pairs = json.load(open(args.gt_file, "r"))

    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(
        os.path.join(args.output_dir, f"{args.output_name}.json"), "w")

    for index, sample in enumerate(tqdm(gt_qa_pairs)):
        task_name = sample["question_type"] 
        video_name = sample["video_name"]
        question_id = sample["question_id"]
        question = sample["question"]
        answer_number = sample["answer_number"]
        candidates = sample["candidates"]
        answer = sample["gt_answer"]
        background9 = sample["9_background"]

        sample_set = {
            "task_name": task_name,
            "question": question,
            "id": question_id,
            "answer_number": answer_number,
            "candidates": candidates,
            "answer": answer,
        }

        video_path = os.path.join(args.video_dir, video_name)
                
        if os.path.exists(video_path):
            output = pos_inference(
                video_path,
                background9,
                question,
                candidates,
                model,
                tokenizer,
                args.num_frames,
                args.pos,
                args.video_dir
            )
            print(output)
            sample_set["pred"] = output
            ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_inference(args)
