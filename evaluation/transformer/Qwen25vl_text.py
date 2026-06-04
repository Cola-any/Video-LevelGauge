#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
import torch
import copy
import numpy as np
from decord import VideoReader, cpu
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

import math

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Directory containing video files.", required=False, default='Qwen25-VL-7B-Instruct')
    parser.add_argument("--video_dir", help="Directory containing video files.", required=False, default='./LevelGauge/videos')
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question and answer.", required=False, default='./LevelGauge/json/Pos_MCQA_1177_final.json')
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=False, default='./qwen25vl_output')
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=False, default='debug')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--pos", type=str, default="10-02")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--bk_frame_number", type=int, default=54)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--rope_scaling_factor", type=int, default=2)
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

def build_pos_content(video_path, background9, background19, pos, gt_frame_number, bk_frame_number, video_dir):
    if pos[:2] == "10":
        pos_num = int(pos[-2:])
        if pos_num == int(pos[:2]):
            content = [
                {
                    "type": "video",
                    "video": video_path,
                    'nframes': gt_frame_number,
                    "resized_height": 600,
                    "resized_width": 800,
                }
            ]
        else:
            pos_num = int(pos[-2:])
            background_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
            background_list.insert(pos_num, video_path)
            content = []
            for video_path_ in background_list:
                if video_path_ == video_path:
                    content.append(
                        {
                            "type": "video",
                            "video": video_path,
                            'nframes': gt_frame_number,
                            "resized_height": 600,
                            "resized_width": 800,
                        }
                    )
                else:
                    text_num = int(video_path_)
                    content.append(
                        {
                            "type": "text",
                            "text": background9[text_num],
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
    tokenizer,
    frame_number,
    bk_frame_number,
    background9,
    background19,
    pos,
    video_dir
):
    candidates_prompt = get_option_prompt(candidates, version='v4')
    prompt = f'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n{candidates_prompt}\n'
    prompt = prompt + "Answer with the option's letter from the given choices directly."
    
    content1 = build_pos_content(video_path, background9, background19, pos, frame_number, bk_frame_number, video_dir)
    content1.append({"type": "text", "text": prompt})
    # content2.append({"type": "text", "text": prompt})

    messages1 = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {
            "role": "user",
            "content": content1,
        }
    ]

    text = processor.apply_chat_template(
        messages1, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages1, return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        # fps=fps_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
        )
    inputs = inputs.to("cuda")

    current_gen_kwargs = {
                "max_new_tokens": 8,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }

    cont = model.generate(
                **inputs,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if current_gen_kwargs["temperature"] > 0 else False,
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=True, # False
            )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
    outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return outputs[0]


def run_inference(args):

    # overwrite_config = {}
    # overwrite_config["mm_newline_position"] = "frame"
    # llava_model_args["overwrite_config"] = overwrite_config
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_dir)
    # Load questions and answers
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    gt_qa_pairs = json.load(open(args.gt_file, "r"))
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(
        os.path.join(args.output_dir, f"{args.output_name}.json"), "w")

    text_context = json.load(open("./background/text_bk_len1400.json", "r"))

    # Iterate over each sample in the ground truth file
    for index, sample in enumerate(tqdm(gt_qa_pairs)):
        task_name = sample["question_type"] # 根据什么划分subset进行评估
        video_name = sample["video_name"]
        question_id = sample["question_id"]
        question = sample["question"]
        answer_number = sample["answer_number"]
        candidates = sample["candidates"]
        answer = sample["gt_answer"]
        text_num = index % len(text_context)
        background9 = text_context[text_num]
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
                tokenizer,
                args.num_frames,
                args.bk_frame_number,
                background9,
                background19,
                args.pos,
                args.video_dir
            )
            output = output.replace("In the image", "In the video")
            # print(output)
            sample_set["pred"] = output
            ans_file.write(json.dumps(sample_set) + "\n")
        else:
            print("###########", video_path)

    ans_file.close()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_inference(args)
