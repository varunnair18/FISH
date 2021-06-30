import os
import numpy as np
import matplotlib.pyplot as plt

def jointly_draw_learning_curves(task, keyword, merge_step):
    folders = [
        f"tmp/{task}_sparse_asgd_merge{merge_step}@0", 
        f"tmp/{task}_fully_normal@0", 
        f"tmp/{task}_fully_asgd_sum_merge{merge_step}@0", 
        f"tmp/{task}_fully_asgd_mean_merge{merge_step}@0", 
    ]

    get_key = lambda name: int(name.split("_")[-1].split(".")[0])

    def extract_loss_n_acc(_file):
        with open(_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.rstrip()
                if keyword in line:
                    acc = float(line.split(" ")[-1])
                if "eval_loss" in line:
                    loss = float(line.split(" ")[-1])

        return acc, loss

    def smooth(values, window_size):
        half_size = window_size // 2

        return [np.mean(values[i - half_size: i + half_size]) for i in range(len(values))]
        

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(10.5, 14.5)

    colors = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:purple"]
    names = [
        "Sparsely updated ASGD model", 
        "Densely updated single model", 
        "Densely updated ASGD model (summation)", 
        "Densely updated ASGD model (average)",
    ]

    for folder, color, name in zip(folders, colors, names):
        files = os.listdir(folder)

        # train_files = [os.path.join(folder, f) for f in files if f.startswith("train_")]
        eval_files = [os.path.join(folder, f) for f in files if f.startswith("eval_")]

        # train_files.sort(key=get_key)
        eval_files.sort(key=get_key)

        iters = [get_key(f) for f in eval_files]

        # train_results = [extract_loss_n_acc(f) for f in train_files]
        eval_results = [extract_loss_n_acc(f) for f in  eval_files]

        # train_accs, train_losses = zip(*train_results)
        eval_accs, eval_losses = zip(*eval_results)

        # train_accs = smooth(train_accs, 5)
        # train_losses = smooth(train_losses, 5)

        # eval_accs = smooth(eval_accs, 5)
        # eval_losses = smooth(eval_losses, 5)
        
        ax1 = axs[0]
        # ax1.plot(iters, train_accs, color=color, label=f"{name} (Train)")
        ax1.plot(iters, eval_accs, color=color, label=f"{name}", linestyle="-")
        label = keyword.split("_")
        label = [l.capitalize() for l in label]
        label = " ".join(label)
        ax1.set_ylabel(label)
        ax1.grid(linewidth=2, linestyle=":")

        ax1.set_title(f"{task.upper()} (merge step={merge_step})")

        ax1.legend()

        ax1 = axs[1]
        # ax1.plot(iters, train_losses, color=color, label=f"{name} (Train)")
        ax1.plot(iters, eval_losses, color=color, label=f"{name}", linestyle="-")

        ax1.grid(linewidth=2, linestyle=":")

        ax1.set_ylabel('Eval Loss')
        ax1.set_xlabel('Iteration')

        ax1.legend()

    plt.savefig(f"img/{task}_learning_curve_merge{merge_step}.png")


def jointly_draw_learning_curves_merge_step(task, keyword):
    
    folders = [f"tmp/{task}_sparse_asgd_merge{merge_step}@0" for merge_step in [10, 30, 100]]

    get_key = lambda name: int(name.split("_")[-1].split(".")[0])

    def extract_loss_n_acc(_file):
        with open(_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.rstrip()
                if keyword in line:
                    acc = float(line.split(" ")[-1])
                if "eval_loss" in line:
                    loss = float(line.split(" ")[-1])

        return acc, loss

    def smooth(values, window_size):
        half_size = window_size // 2

        return [np.mean(values[i - half_size: i + half_size]) for i in range(len(values))]
        

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(10.5, 14.5)

    colors = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:purple"]
    names = [
        "Sparsely updated ASGD model (merge step=10)", 
        "Sparsely updated ASGD model (merge step=30)", 
        "Sparsely updated ASGD model (merge step=100)", 
    ]

    for folder, color, name in zip(folders, colors, names):
        files = os.listdir(folder)

        # train_files = [os.path.join(folder, f) for f in files if f.startswith("train_")]
        eval_files = [os.path.join(folder, f) for f in files if f.startswith("eval_")]

        # train_files.sort(key=get_key)
        eval_files.sort(key=get_key)

        iters = [get_key(f) for f in eval_files]

        # train_results = [extract_loss_n_acc(f) for f in train_files]
        eval_results = [extract_loss_n_acc(f) for f in  eval_files]

        # train_accs, train_losses = zip(*train_results)
        eval_accs, eval_losses = zip(*eval_results)

        # train_accs = smooth(train_accs, 5)
        # train_losses = smooth(train_losses, 5)

        # eval_accs = smooth(eval_accs, 5)
        # eval_losses = smooth(eval_losses, 5)
        
        ax1 = axs[0]
        # ax1.plot(iters, train_accs, color=color, label=f"{name} (Train)")
        ax1.plot(iters, eval_accs, color=color, label=f"{name}", linestyle="-")
        label = keyword.split("_")
        label = [l.capitalize() for l in label]
        label = " ".join(label)
        ax1.set_ylabel(label)
        ax1.grid(linewidth=2, linestyle=":")

        ax1.set_title(f"{task.upper()}")

        ax1.legend()

        ax1 = axs[1]
        # ax1.plot(iters, train_losses, color=color, label=f"{name} (Train)")
        ax1.plot(iters, eval_losses, color=color, label=f"{name}", linestyle="-")

        ax1.grid(linewidth=2, linestyle=":")

        ax1.set_ylabel('Eval Loss')
        ax1.set_xlabel('Iteration')

        ax1.legend()

    plt.savefig(f"img/{task}_compare_merge_step.png")


if __name__ == "__main__":

    for task in ["rte", "cola", "mrpc", "stsb"]:
        if task in ["rte"]:
            keyword = "eval_accuracy"
        elif task in ["cola"]:
            keyword = "eval_matthews_correlation"
        elif task in ["mrpc", "stsb"]:
            keyword = "eval_combined_score"

        for merge_step in [10, 30, 100]:
            jointly_draw_learning_curves(task, keyword, merge_step)

        jointly_draw_learning_curves_merge_step(task, keyword)