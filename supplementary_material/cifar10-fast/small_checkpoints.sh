num_samples=1024
epochs=100
lr=0.4

for keep_ratio in 0.005 0.02 0.1
do
    for lr in 0.4 0.2 0.08 0.04 0.02
    do
    python su_dawn.py \
            --num_samples $num_samples \
            --keep_ratio $keep_ratio \
            --epochs $epochs \
            --lr $lr \
            --save_file su_s${num_samples}_k${keep_ratio}_e${epochs}_l${lr}.tsv
    done
done

for keep_ratio in 0.005 0.02 0.1
do
    for lr in 0.4 0.2 0.08 0.04 0.02
    do
        python su_dawn.py \
                --num_samples $num_samples \
                --keep_ratio $keep_ratio \
                --epochs $epochs \
                --lr $lr \
                --fix_mask \
                --save_file su_s${num_samples}_k${keep_ratio}_e${epochs}_l${lr}_fix.tsv
    done
done


for keep_ratio in 0.005 0.02 0.1
do
    for lr in 0.4 0.2 0.08 0.04 0.02
    do
    python su_dawn.py \
            --mask_method "random" \
            --num_samples $num_samples \
            --keep_ratio $keep_ratio \
            --epochs $epochs \
            --lr $lr \
            --save_file su_random_k${keep_ratio}_e${epochs}_l${lr}.tsv
    done
done

for keep_ratio in 0.005 0.02 0.1
do
    for lr in 0.4 0.2 0.08 0.04 0.02
    do
        python su_dawn.py \
                --mask_method "random" \
                --num_samples $num_samples \
                --keep_ratio $keep_ratio \
                --epochs $epochs \
                --lr $lr \
                --fix_mask \
                --save_file su_random_k${keep_ratio}_e${epochs}_l${lr}_fix.tsv
    done
done