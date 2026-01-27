import argparse
import os
import json
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import math

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Directory containing video files.", required=False, default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument("--video_dir", help="Directory containing video files.", required=False, default='./LevelGauge/videos')
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question and answer.", required=False, default='./LevelGauge/json/Pos_MCQA_300_final.json')
    parser.add_argument("--output_dir", help="Directory to save the model response.", required=False, default='./qwen3vl_output')
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=False, default='debug')
    parser.add_argument("--pos", type=str, help="10-00 represents that probe is inserted in the first position, and 10-10 represents probe input only.", default="10-00")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=6)
    return parser.parse_args()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

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

def build_pos_content(video_path, background9, background19, pos, frame_number, video_dir):
    if pos[:2] == "10":
        pos_num = int(pos[-2:])
        if pos_num == int(pos[:2]):
            content = [
                {
                    "type": "video",
                    "video": video_path,
                    'nframes': frame_number,
                    "resized_height": 600,
                    "resized_width": 800,
                }
            ]
        else:
            background_list = list(background9.keys())
            background_list_ = [video_dir + "/" + s for s in background_list]
            background_list_.insert(pos_num, video_path)
            content = []
            for video_path_ in background_list_:
                content.append(
                    {
                        "type": "video",
                        "video": video_path_,
                        'nframes': frame_number,
                        "resized_height": 600,
                        "resized_width": 800,
                    }
                )
        return content
    elif pos[:2] == "20":
        pos_num = int(pos[-2:])
        if pos_num == int(pos[:2]):
            content = [
                {
                    "type": "video",
                    "video": video_path,
                    'nframes': frame_number,
                    "resized_height": 600,
                    "resized_width": 800,
                }
            ]
        else:
            pos_num = int(pos[-2:])
            background_list = list(background19.keys())
            background_list_ = [video_dir + "/" + s for s in background_list]
            background_list_.insert(pos_num, video_path)
            content = []
            for video_path_ in background_list_:
                content.append(
                    {
                        "type": "video",
                        "video": video_path_,
                        'nframes': frame_number,
                        "resized_height": 600,
                        "resized_width": 800,
                    }
                )
        return content
    else:
        print("error")

def pos_inference(
    video_path,
    question,
    candidates,
    model,
    processor,
    frame_number,
    background9,
    background19,
    pos,
    video_dir
):
    candidates_prompt = get_option_prompt(candidates, version='v4')
    prompt = f'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n{candidates_prompt}\n'
    prompt = prompt + "Answer with the option's letter from the given choices directly."
    
    content = build_pos_content(video_path, background9, background19, pos, frame_number, video_dir)
    content.append({"type": "text", "text": prompt})

    if frame_number != 0:
        messages = [
        {
            "role": "user",
            "content": content,
        }
        ]
    else:
        messages = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
        ]
    # print(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, 
                                                                   image_patch_size= 16,
                                                                   return_video_metadata=True)
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=8)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def run_inference(args):

    model_path = args.model_dir #  The following output example is from a tiny test model
    processor = AutoProcessor.from_pretrained(model_path)

    model, output_loading_info = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2", output_loading_info=True)

    gt_qa_pairs = json.load(open(args.gt_file, "r"))
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(
        os.path.join(args.output_dir, f"{args.output_name}.json"), "w")

    # Iterate over each sample in the ground truth file
    for index, sample in enumerate(tqdm(gt_qa_pairs)):
        task_name = sample["question_type"] # 
        video_name = sample["video_name"]
        question_id = sample["question_id"]
        question = sample["question"]
        answer_number = sample["answer_number"]
        candidates = sample["candidates"]
        answer = sample["gt_answer"]
        background9 = sample["9_background"]
        background19 = sample["19_background"]

        sample_set = {
            "task_name": task_name,
            "question": question,
            "id": question_id,
            "answer_number": answer_number,
            "candidates": candidates,
            "answer": answer,
        }

        # Load video
        video_path = os.path.join(args.video_dir, video_name)
                
        if os.path.exists(video_path):
            output = pos_inference(
                video_path,
                question,
                candidates,
                model,
                processor,
                args.num_frames,
                background9,
                background19,
                args.pos,
                args.video_dir
            )
            output = output.replace("In the image", "In the video")
            print(output)
            sample_set["pred"] = output
            ans_file.write(json.dumps(sample_set) + "\n")
        else:
            print("###########", video_path)

    ans_file.close()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_inference(args)
