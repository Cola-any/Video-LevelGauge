import argparse
import os
import json
from tqdm import tqdm
import numpy as np
from qwen_vl_utils import process_vision_info
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8005/v1" 

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory containing video files.", required=False, default='./LevelGauge/videos')
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question and answer.", required=False, default='./LevelGauge/json/Pos_MCQA_300_final.json')
    parser.add_argument("--output_dir", help="Directory to save the model response.", required=False, default='./mimovl_output')
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=False, default='10-00')
    parser.add_argument("--pos", type=str, help="10-00 represents that probe is inserted in the first position, and 10-10 represents probe input only.", default="10-00")
    parser.add_argument("--num_frames", type=int, default=6)
    return parser.parse_args()

# https://github.com/QwenLM/Qwen2.5-VL#start-an-openai-api-service
def prepare_message_for_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}

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

def build_pos_content(video_path, background9, pos, frame_number, video_dir):
    if pos[:2] == "10":
        pos_num = int(pos[-2:])
        #---------------- probe input only
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
        #---------------- insert probe into the background
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
    else:
        print("error")

def pos_inference(
    video_path,
    question,
    candidates,
    frame_number,
    background9,
    pos,
    video_dir
):
    candidates_prompt = get_option_prompt(candidates, version='v4')
    prompt = f'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n{candidates_prompt}\n'
    prompt = prompt + "Answer with the option's letter from the given choices directly. /no_think"
    
    content = build_pos_content(video_path, background9, pos, frame_number, video_dir)
    # print(content)
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {
            "role": "user",
            "content": content,
        }
    ]
    video_messages, _ = prepare_message_for_vllm(messages)

    try:
        chat_response = client.chat.completions.create(
                model="MiMo-VL",
                messages=video_messages,
                temperature=0.0,
                max_tokens = 256
            )
        return chat_response.choices[0].message.content
    except:
        return "###failed_response"

def run_inference(args):

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
                question,
                candidates,
                args.num_frames,
                background9,
                args.pos,
                args.video_dir
            )
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
