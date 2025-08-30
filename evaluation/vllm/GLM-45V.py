import argparse
import os
import json
from tqdm import tqdm
from openai import OpenAI
import json

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
    parser.add_argument("--output_dir", help="Directory to save the model response.", required=False, default='./glm45v_output')
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=False, default='10-00')
    parser.add_argument("--pos", type=str, help="10-00 represents that probe is inserted in the first position, and 10-10 represents probe input only.", default="10-00")
    parser.add_argument("--num_frames", type=int, default=6)
    return parser.parse_args()

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
    video_name,
    question,
    candidates,
    background9,
    pos,
):
    candidates_prompt = get_option_prompt(candidates, version='v4')
    prompt = f'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n{candidates_prompt}\n'
    prompt = prompt + "Answer with the option's letter from the given choices directly. /nothink" #  
    pos_num = int(pos[-2:])
    # print(pos_num)
    #---------------- probe input only
    if pos_num == int(pos[:2]):
        video_name = video_name.split(".")[0]
        frames_dir = "./LevelGauge/frame/" + video_name
        frame_paths = []
        for i in range(6):
            frame_paths.append(frames_dir+"/"+f"frame{i}.jpg")
        contents = []
        for frame_path in frame_paths:
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"file:{frame_path}"
                    },
                },
            )
        contents.append({"type": "text", "text": prompt})
    #---------------- insert probe into the background
    else:
        background_list = list(background9.keys())
        background_list.insert(pos_num, video_name)
        # print(background_list)
        contents = []
        for video_name_ in background_list:
            video_name_ = video_name_.split(".")[0]
            frames_dir = "./LevelGauge/frame/" + video_name_
            frame_paths = []
            for i in range(6):
                frame_paths.append(frames_dir+"/"+f"frame{i}.jpg")
            for frame_path in frame_paths:
                contents.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"file:{frame_path}"
                        },
                    },
                )
        contents.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": contents,
        },
    ]
    try:
        chat_response = client.chat.completions.create(
            model="glm-4.5v",
            messages=messages,
            temperature=0.0,
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
                video_name,
                question,
                candidates,
                background9,
                args.pos,
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
