epochs=50

for keep_ratio in 0.1
do
    for merge_steps in 10 30 100
    do
        python asgd_dawn.py \
            --mask_method "random" \
            --keep_ratio $keep_ratio \
            --epochs $epochs \
            --merge_steps $merge_steps \
            --lr 0.08 \
            --save_file asgd_random_m${merge_steps}_k${keep_ratio}_e${epochs}.tsv

    done
done