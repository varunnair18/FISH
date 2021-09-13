import numpy as np

def get_acc(file_name):
    try:
        with open(file_name, "r") as f:
            line = f.readlines()[-1]

            acc = float(line.split("\t")[-1])
    except:
        acc = "-"

    return acc


def cumsom_avg_std(input_list):

    available_list = [acc for acc in input_list if acc != "-"]

    if len(available_list) == 0:
        return "-"

    mean = np.mean(available_list)
    std = np.std(available_list)

    return f"{mean:.2f}({std:.2f})"


files = []

xs = ["0.8", "0.4", "0.2", "0.08", "0.04", "0.02"]
ys = ["0.02", "0.1", "1.0"]

row_format = "{:>15}" * (len(xs) + 1)

for merging_steps in [100, 500, 2500, 3000]:
    print(f"merging_steps = {merging_steps}")
    
    print(row_format.format("-", *xs))

    for sparsity in ys:
        acc_list = []
        for lr in xs:
            seed_list = []
            for seed in [0, 1, 2, 3, 4]:

                # if lr == "0.4":
                #     file_name = f"logs/su_s1024_k{sparsity}_e{epoch}.tsv"
                # else:

                if sparsity == "1.0":
                    file_name = f"logs/asgd_resnet34_m{merging_steps}_k{sparsity}_e50_l{lr}_s2@{seed}.tsv"

                else:
                    file_name = f"logs/asgd_same_resnet34_m{merging_steps}_k{sparsity}_e50_l{lr}_r1_w5_s2@{seed}.tsv"

                # print(file_name)

                acc = get_acc(file_name)

                seed_list.append(acc)

            acc = cumsom_avg_std(seed_list)
            acc_list.append(acc)

        print(row_format.format(sparsity, *acc_list))
