import re
import sys
import torch
import matplotlib.pyplot as plt


def draw_mask_distribution(file_name, save_name):
    mask_dict = torch.load(file_name)

    layers_names = [
        ("bert.embeddings", None), 
        ("bert.encoder.layer", int), 
        ("bert.pooler", None), 
        ("classifier", None)
    ]

    now_layer_idx = 0
    now_sub_layer_idx = 0
    
    ratio_list = []

    last_layer_idx = -1
    last_sub_layer_idx = -1
    for name, mask in mask_dict.items():
        while layers_names[now_layer_idx][0] not in name:
            now_layer_idx += 1
            now_sub_layer_idx = 0
        # Find the corresponding layer
        if layers_names[now_layer_idx][1] is int:
            while layers_names[now_layer_idx][0] + f".{now_sub_layer_idx}" not in name:
                now_sub_layer_idx += 1

        if (now_layer_idx, now_sub_layer_idx) != (last_layer_idx, last_sub_layer_idx):
            # found new layer, append the ratio list to store it
            ratio_list.append([])
            last_layer_idx = now_layer_idx
            last_sub_layer_idx = now_sub_layer_idx

        ratio_list[-1].append(
            [mask.sum().item(), torch.prod(torch.tensor(mask.shape)).item()]
        )

    # aggregate parameters belong to the same layer

    for i in range(len(ratio_list)):
        trainable_p = sum([p[0] for p in ratio_list[i]])
        all_p = sum([p[1] for p in ratio_list[i]])

        ratio_list[i] = trainable_p / all_p

    # exclude the classification layer, because it is always 100% trainable
    ratio_list.pop(-1)

    plt.bar(range(len(ratio_list)), ratio_list, align='center')
    plt.xlabel("Layer index (smaller index is closed to inputs)")
    plt.ylabel("Ratio of trainable parameters")
    plt.title("Mask distribution")

    plt.grid(linestyle="--", alpha=0.5)

    if save_name is None:
        save_name = "mask_distribution.png"

    plt.savefig(save_name)


def draw_val_curve(file_name, save_name=None):
    with open(file_name, "r") as f:
        lines = f.readlines()

    def extract_values(lines, keywords, notin_keywords, target_tuple):
        for line in lines:
            conditions = [word in line for word in keywords] + \
                         [word not in line for word in notin_keywords]
            if all(conditions):
                line = re.split("[\[\], =()]+", line)

                for i, word in enumerate(line):
                    for keyword, _type, _list in target_tuple:
                        if word == keyword:
                            # next word is the value
                            _list.append(_type(line[i + 1]))

        return target_tuple

    def get_baselines(lines):
        num_samples = []
        eval_accs = []
        eval_losses = []

        keywords = [
            "[classifier all trained]",
            "num_samples = ", 
            "epoch = 3.0"
        ]
        notin_keywords = [
            "lr"
        ]

        keywords_type_list_tuple = [
                ("num_samples", int, num_samples),
                ("eval_accuracy", float, eval_accs),
                ("eval_loss", float, eval_losses),
            ]

        extract_values(lines, keywords, notin_keywords, keywords_type_list_tuple)

        assert len(num_samples) == len(eval_accs) == len(eval_losses)

        return num_samples, eval_accs, eval_losses

    def get_baselines_lr(lines, value):
        num_samples = []
        eval_accs = []
        eval_losses = []

        keywords = [
            "[classifier all trained, lr=1e-4]",
            "num_samples = ", 
            "epoch = 3.0"
        ]
        notin_keywords = []

        keywords_type_list_tuple = [
            ("num_samples", int, num_samples),
            ("eval_accuracy", float, eval_accs),
            ("eval_loss", float, eval_losses),
        ]

        extract_values(lines, keywords, notin_keywords, keywords_type_list_tuple)
                
        assert len(num_samples) == len(eval_accs) == len(eval_losses)

        return num_samples, eval_accs, eval_losses

    def get_baselines_epoch(lines, value):
        num_samples = []
        eval_accs = []
        eval_losses = []

        keywords = [
            "[classifier all trained, lr=1e-4]",
            "num_samples = ", 
            "epoch = 7.0"
        ]
        notin_keywords = []

        keywords_type_list_tuple = [
            ("num_samples", int, num_samples),
            ("eval_accuracy", float, eval_accs),
            ("eval_loss", float, eval_losses),
        ]

        extract_values(lines, keywords, notin_keywords, keywords_type_list_tuple)
                
        assert len(num_samples) == len(eval_accs) == len(eval_losses)

        return num_samples, eval_accs, eval_losses

    def get_bitfit(lines):
        num_samples = []
        eval_accs = []
        eval_losses = []

        keywords = [
            "bitfit (lr=1e-3), epoch = 7.0"
        ]

        notin_keywords = []

        keywords_type_list_tuple = [
            ("num_samples", int, num_samples),
            ("eval_accuracy", float, eval_accs),
            ("eval_loss", float, eval_losses),
        ]
        
        extract_values(lines, keywords, notin_keywords, keywords_type_list_tuple)

        assert len(eval_accs) == len(eval_losses)

        return num_samples, eval_accs, eval_losses

    def get_random(lines):
        num_samples = []
        eval_accs = []
        eval_losses = []

        keywords = [
            "random mask, epoch = 7.0"
        ]

        notin_keywords = []

        keywords_type_list_tuple = [
            ("num_samples", int, num_samples),
            ("eval_accuracy", float, eval_accs),
            ("eval_loss", float, eval_losses),
        ]
        
        extract_values(lines, keywords, notin_keywords, keywords_type_list_tuple)

        assert len(eval_accs) == len(eval_losses)

        return num_samples, eval_accs, eval_losses

    def get_normal_train(lines):
        num_samples = []
        eval_accs = []
        eval_losses = []

        keywords = [
            "normal training"
        ]

        notin_keywords = []

        keywords_type_list_tuple = [
            ("num_samples", int, num_samples),
            ("eval_accuracy", float, eval_accs),
            ("eval_loss", float, eval_losses),
        ]
        
        extract_values(lines, keywords, notin_keywords, keywords_type_list_tuple)

        assert len(eval_accs) == len(eval_losses)

        return num_samples, eval_accs, eval_losses

    def get_diff_prune(lines):
        return 0, 0.9312, 0

    baselines_x, baselines_y, baselines_y2 = get_baselines(lines)
    # lr_minusthree_x, lr_minusthree_y, _ = get_baselines_lr(lines, 1e-3)
    lr_minusfour_x, lr_minusfour_y, lr_minusfour_y2 = get_baselines_lr(lines, None)
    epoch7_x, epoch7_y, epoch7_y2 = get_baselines_epoch(lines, None)
    _, bitfit_y, bitfit_y2 = get_bitfit(lines)    
    _, normal_train_y, normal_train_y2 = get_normal_train(lines)
    _, diff_prune_y, diff_prune_y2 = get_diff_prune(lines)
    _, random_y, random_y2 = get_random(lines)

    plt.clf()
    plt.figure(figsize=(7,10))
    ax = plt.subplot(2, 1, 1)
    plt.plot(baselines_x, baselines_y, label="lr=2e-5, epoch=3")
    plt.plot(lr_minusfour_x, lr_minusfour_y, label="lr=1e-4, epoch=3")
    plt.plot(epoch7_x, epoch7_y, label="lr=1e-4, epoch=7")
    plt.hlines(normal_train_y, xmin=baselines_x[0], xmax=baselines_x[-1], linestyles="solid", color="black", label="Normally trained")
    plt.hlines(bitfit_y, xmin=baselines_x[0], xmax=baselines_x[-1], linestyles="dashed", colors="b", label="BitFit")
    plt.hlines(diff_prune_y, xmin=baselines_x[0], xmax=baselines_x[-1], linestyles="dashed", label="DiffPrune")

    plt.hlines(random_y, xmin=baselines_x[0], xmax=baselines_x[-1], linestyles="dashed", color="g", label="Random (epoch=7)")

    plt.legend(loc="best")

    # plt.xlabel("Number of samples")
    plt.ylabel("Evaluation accuracy")
    plt.title("Methods Comparison on SST-2")

    plt.grid(linestyle="--", alpha=0.5)

    plt.subplot(2, 1, 2)

    plt.plot(baselines_x, baselines_y2, label="lr=2e-5")
    plt.plot(lr_minusfour_x, lr_minusfour_y2, label="lr=1e-4, epoch=3")
    plt.plot(epoch7_x, epoch7_y2, label="lr=1e-4, epoch=7")
    plt.hlines(normal_train_y2, xmin=baselines_x[0], xmax=baselines_x[-1], linestyles="solid", color="black", label="Normally trained")
    plt.hlines(bitfit_y2, xmin=baselines_x[0], xmax=baselines_x[-1], linestyles="dashed", colors="b", label="BitFit")
    plt.hlines(random_y2, xmin=baselines_x[0], xmax=baselines_x[-1], linestyles="dashed", color="g", label="Random (epoch=7)")

    plt.legend(loc="best")

    plt.xlabel("Number of samples")
    plt.ylabel("Evaluation loss")

    plt.setp(ax.get_xticklabels(), visible=False)

    plt.grid(linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_name is None:
        save_name = "method_comparison.png"

    plt.savefig(save_name)


if __name__ == "__main__":
    task = sys.argv[1] # draw task to use
    file_name = sys.argv[2]
    save_name = sys.argv[3]

    method = eval(task)

    method(file_name, save_name)