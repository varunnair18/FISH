num_samples=256
split=$1
epochs=$2

for keep_ratio in 0.005 0.02 0.1
do
    for merge_steps in 10 30 100
    do
        for lr in 0.4 0.2 0.08 0.04 0.02
        do
            python asgd_dawn.py \
                --num_samples $num_samples \
                --keep_ratio $keep_ratio \
                --epochs $epochs \
                --merge_steps $merge_steps \
                --lr $lr \
                --split $split \
                --save_file logs/asgd_m${merge_steps}_k${keep_ratio}_e${epochs}_l${lr}_s${split}.tsv
        done
    done
done


for keep_ratio in 0.005 0.02 0.1
do
    for merge_steps in 10 30 100
    do
        for lr in 0.4 0.2 0.08 0.04 0.02
        do
        python asgd_dawn.py \
            --mask_method "random" \
            --keep_ratio $keep_ratio \
            --epochs $epochs \
            --merge_steps $merge_steps \
            --lr $lr \
            --split $split \
            --save_file logs/asgd_random_m${merge_steps}_k${keep_ratio}_e${epochs}_l${lr}_s${split}.tsv
        done
    done
done


# Densely-updated
for merge_steps in 10 30 100
do
    for lr in 0.4 0.2 0.08 0.04 0.02
    do
        python asgd_dawn.py \
            --epochs $epochs \
            --merge_steps $merge_steps \
            --mask_method "all_ones" \
            --lr $lr \
            --split $split \
            --save_file logs/asgd_m${merge_steps}_k1.0_e${epochs}_l${lr}_s${split}.tsv
    done
done


