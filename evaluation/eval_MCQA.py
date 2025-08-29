import argparse
from tqdm import tqdm
import numpy as np
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", default="./output/internvl_answer/10-00.json", help="The path to file containing prediction.")
    parser.add_argument("--pos", type=str, default="10-00")
    parser.add_argument("--output_dir", type=str, default="./output/internvl_acc")
    parser.add_argument("--model_name", type=str, default="internvl3")
    args = parser.parse_args()
    return args

def map_prediction_to_option(pred, model_name):
    pred_option = "none"
    if isinstance(pred, str):
        if model_name == "mimovl": 
            pred = pred.replace("<think>\n</think>\n\n","")
            prediction_letter = pred[0]
        elif model_name == "glm45v":
            pred = pred.split("<|begin_of_box|>")[-1]
            prediction_letter = pred[0]
        else:
            prediction_letter = pred[0]
        if prediction_letter in "abcdeABCDE":
            pred_option = prediction_letter.lower()
        if "answer is " in pred:
            pred = pred[pred.index("answer is"):]
        if "A:" in pred or "A)" in pred:
            pred_option = "a"
        elif "B:" in pred or "B)" in pred:
            pred_option = "b"
        elif "C:" in pred or "C)" in pred:
            pred_option = "c"
        elif "D:" in pred or "D)" in pred:
            pred_option = "d"
        elif "E:" in pred or "E)" in pred:
            pred_option = "e"
    return pred_option


def check_ans(pred, gt, model_name):
    flag = False

    pred_option = map_prediction_to_option(pred, model_name)

    if pred_option not in "abcde":
        print(f"Model does not follow the instruction: {pred}")
    elif pred_option == gt.lower():
        flag = True

    return flag


def main():
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output_dir + "/" + args.pos + ".json"

    task_accuracy = {}
    example_num = 0
    for new_pred_content in tqdm(new_pred_contents):
        task_name = new_pred_content["task_name"]
        if task_name not in task_accuracy:
            task_accuracy[task_name] = {
                "yes_count": 0,
                "no_count": 0,
            }
        example_num = example_num + 1
        gt = chr(ord("a") + new_pred_content["answer_number"]) 
        if check_ans(
            pred=new_pred_content["pred"],
            gt=gt,
            model_name=args.model_name,
        ):
            task_accuracy[task_name]["yes_count"] += 1
        else:
            task_accuracy[task_name]["no_count"] += 1

    accuracy_list = []
    for task_name in task_accuracy:
        yes_count = task_accuracy[task_name]["yes_count"]
        no_count = task_accuracy[task_name]["no_count"]
        accuracy = yes_count / (yes_count + no_count)
        task_accuracy[task_name]["accuracy"] = accuracy
        accuracy_list.append(accuracy)
        print(task_name)
        print("\tYes count:", yes_count)
        print("\tNo count:", no_count)
        print("\tAccuracy:", accuracy)
    print("Total examples:", example_num)
    task_accuracy["Total examples"] = example_num
    if len(accuracy_list) > 1:
        print("Average accuracy:", np.mean(accuracy_list))
        task_accuracy["Average accuracy"] = np.mean(accuracy_list)
    
    with open(output_path, "w") as f:
        json.dump(task_accuracy, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
