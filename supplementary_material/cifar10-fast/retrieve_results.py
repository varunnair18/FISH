def get_acc(file_name):
    try:
        with open(file_name, "r") as f:
            line = f.readlines()[-1]

            acc = float(line.split("\t")[-1])
    except:
        acc = "-"

    return acc

# files = []

# row_format = "{:>10}" * 4

# for epoch in [24, 40, 60, 100]:
#     print(f"epoch = {epoch}")
#     print(row_format.format("-", 256, 1024, 4096))
#     for sparsity in ["0.005", "0.02", "0.10"]:
#         acc_list = []
#         for num_samples in [256, 1024, 4096]:
#             file_name = f"su_s{num_samples}_k{sparsity}_e{epoch}_fix.tsv"

#             acc = get_acc(file_name)

#             acc_list.append(acc)

#         print(row_format.format(sparsity, *acc_list))


# files = []

# row_format = "{:>10}" * 4

# for epoch in [12, 20, 30, 50]:
#     print(f"epoch = {epoch}")
#     print(row_format.format("-", 10, 30, 100))

#     for sparsity in ["0.005", "0.02", "0.1", "fully"]:
#         acc_list = []
        
#         for merging_steps in [10, 30, 100]:

#             file_name = f"asgd_random_m{merging_steps}_k{sparsity}_e{epoch}.tsv"

#             acc = get_acc(file_name)
#             acc_list.append(acc)

#         print(row_format.format(sparsity, *acc_list))


# files = []

# xs = ["0.4", "0.2", "0.08", "0.04", "0.02"]
# ys = ["0.005", "0.02", "0.1"]

# row_format = "{:>15}" * (len(xs) + 1)

# for epoch in [100]:
#     print(f"epoch = {epoch}")
    
#     print(row_format.format("-", *xs))

#     for sparsity in ys:
#         acc_list = []
#         for lr in xs:
#             # if lr == "0.4":
#             #     file_name = f"logs/su_s1024_k{sparsity}_e{epoch}.tsv"
#             # else:
#             file_name = f"su_random_k{sparsity}_e{epoch}_l{lr}_fix.tsv"

#             acc = get_acc(file_name)

#             acc_list.append(acc)

#         print(row_format.format(sparsity, *acc_list))

files = []

xs = ["0.4", "0.2", "0.08", "0.04", "0.02"]
ys = ["0.005", "0.02", "0.1"]

row_format = "{:>15}" * (len(xs) + 1)

for merging_steps in [10, 30, 100]:
    print(f"merging_steps = {merging_steps}")
    
    print(row_format.format("-", *xs))

    for sparsity in ys:
        acc_list = []
        for lr in xs:
            # if lr == "0.4":
            #     file_name = f"logs/su_s1024_k{sparsity}_e{epoch}.tsv"
            # else:
            file_name = f"asgd_m{merging_steps}_k{sparsity}_e50_l{lr}.tsv"

            acc = get_acc(file_name)

            acc_list.append(acc)

        print(row_format.format(sparsity, *acc_list))



