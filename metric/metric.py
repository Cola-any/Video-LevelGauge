import os
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc_dir", required=False, default='./output/example_acc')
    return parser.parse_args()

def classify_by_trend(y):
    x = np.arange(len(y))
    
    # first-order linear fit
    coeffs_linear = np.polyfit(x, y, deg=1)
    y_linear_fit = np.polyval(coeffs_linear, x)
    residual_linear = np.mean((y - y_linear_fit) ** 2)
    slope = coeffs_linear[0]

    # second-order fit
    coeffs_quad = np.polyfit(x, y, deg=2)
    y_quad_fit = np.polyval(coeffs_quad, x)
    residual_quad = np.mean((y - y_quad_fit) ** 2)

    print(f'First-order linear fit slope: {slope:.15f}')
    print(f'First-order linear fit residual: {residual_linear:.15f}')
    print(f'Second-order fit residual: {residual_quad:.15f}')
    
    if residual_linear <= 3:
        if abs(slope) <= 0.5:
            return "Stable"
        elif slope > 0.5:
            return "Neighbor preference"
        elif slope < -0.5:
            return "Head preference"
    elif residual_linear > 3 and residual_quad <= 2:
        return "U-shaped"
    else:
        return "Volatile"

def compute_and_print_base_and_metrics(folder_path):
    fields = [
        "Spatial Relationship",
        "Optical Character Recognition",
        "Object Recognition/Reasoning",
        "Action Recognition/Reasoning",
        "Attribute Perception",
        "Count Problem",
        "Average accuracy"
    ]
    
    base_file = "10-10.json" # accuracy of probe input only
    with open(os.path.join(folder_path, base_file), "r") as f:
        base_data = json.load(f)
        base_accuracy_dict = {
            field: base_data[field] if field == "Average accuracy" else base_data[field]["accuracy"]
            for field in fields
        }

    rs_dict = {field: [] for field in fields}

    json_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".json") and f != base_file
    ])
    
    for file in json_files:
        with open(os.path.join(folder_path, file), "r") as f:
            data = json.load(f)
            for field in fields:
                acc_val = data[field] if field == "Average accuracy" else data[field]["accuracy"]
                base_acc = base_accuracy_dict[field]
                rs = (acc_val / base_acc) * 100 if base_acc != 0 else 0
                rs_dict[field].append(rs)

    for field in fields:
        base_acc_pct = base_accuracy_dict[field] * 100
        rs_list = rs_dict[field]

        pbs = np.max(rs_list)
        pran = np.max(rs_list) - np.min(rs_list)
        pvar = np.var(rs_list)
        avg_rs = np.mean(rs_list)
        if field == "Average accuracy":
            cm = 4.0*pran + 3.0*(100.0-avg_rs) + 3.0*pvar

        print(f"=== {field} ===")
        print(f"Pbs : {pbs:.1f}%")
        print(f"Pran: {pran:.1f}%")
        print(f"Pvar: {pvar:.1f}%")
        print(f"Pmean: {avg_rs:.1f}%")
        print(f"Meta Accuracy: {base_acc_pct:.1f}%")
        
    avg_rs_list = rs_dict["Average accuracy"]
    # avg_rs_list = [x / 100 for x in avg_rs_list]
    # print(f"Average accuracy:", avg_rs_list)
    print(f"Composite Metri (CM):", cm)
    print("========= Morphological Recognition ========")
    print(f'MR:', classify_by_trend(avg_rs_list))

if __name__ == "__main__":
    args = parse_args()
    compute_and_print_base_and_metrics(args.acc_dir)