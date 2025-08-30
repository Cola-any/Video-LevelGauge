import json
import matplotlib.pyplot as plt
import numpy as np

def get_relative_score(conetxt_files, probe_file, file_dir, NORM):
    proble_path = file_dir + probe_file + ".json"
    proble_acc = json.load(open(proble_path, "r"))
    assert proble_acc["Total examples"] == 1177
    avg_acc = []
    for contex_file in conetxt_files:
        contex_path = file_dir + contex_file + ".json"
        contex_acc = json.load(open(contex_path, "r"))
        avg_acc.append(contex_acc["Average accuracy"])

    if NORM == True:
        avg_acc = [x / proble_acc["Average accuracy"] for x in avg_acc]
    
    return avg_acc

def plot_pos_bias(probe_file, probe_dir, output_path):
    conetxt_file = ["10-00", "10-01", "10-02", "10-03", "10-04", "10-05", "10-06", "10-07", "10-08", "10-09"]
    avg_acc = get_relative_score(conetxt_file, probe_file, probe_dir, True)
    # print(avg_acc)
    context_10 = avg_acc
    x_axs_10 = [0,1,2,3,4,5,6,7,8,9]

    x1 = [i + 1 for i in x_axs_10]  
    y1 = [v * 100 for v in context_10]  

    plt.figure(figsize=(10, 6))

    plt.plot(x1, y1, color='red', linestyle='-.', marker='o', label='RS')

    yticks = np.arange(80, 101, 2)
    plt.yticks(yticks, [f"{y:.1f}" for y in yticks], fontsize=20)
    plt.xticks(np.arange(1, 11, 1), fontsize=20)  

    plt.title('Model', fontsize=40)
    plt.xlabel('Position of Probes in the Context', fontsize=30)
    plt.ylabel('Relative Score', fontsize=30)
    plt.legend(fontsize=25, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # plt.savefig('./bias_normed.png', dpi=300)
    plt.savefig(output_path)
    plt.show()
    print(f"Save to: ", output_path)

if __name__ == "__main__":
    probe_file = "10-10" # probe input only, we use 10-10 to respresent this
    acc_dir = "./output/example_acc/"
    output_path = './bias_normed.pdf'
    plot_pos_bias(probe_file, acc_dir, output_path)