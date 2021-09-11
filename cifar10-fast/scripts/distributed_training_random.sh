keep_ratio=$1
epochs=$2
merge_steps=$3
lr=$4
split=$5

DIR=cifar10-fast/

python ${DIR}/asgd_dawn.py \
    --mask_method "random" \
    --keep_ratio $keep_ratio \
    --epochs $epochs \
    --merge_steps $merge_steps \
    --lr $lr \
    --split $split \
    --warmup_epoch 5 \
    --same_mask \
    --save_file logs/asgd_random_m${merge_steps}_k${keep_ratio}_e${epochs}_l${lr}_s${split}_w5.tsv