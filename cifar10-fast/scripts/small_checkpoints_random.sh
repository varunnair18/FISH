keep_ratio=$1
epochs=$2
lr=$3
fix_mask=$4

DIR=cifar10-fast

if [ $fix_mask == 1 ]; then
    echo "Fix the mask during training"
    python ${DIR}/su_dawn.py \
        --mask_method "random" \
        --keep_ratio $keep_ratio \
        --epochs $epochs \
        --lr $lr \
        --fix_mask \
        --save_file su_random_k${keep_ratio}_e${epochs}_l${lr}_fix.tsv
else
    echo "The mask is allowed to be updated every epoch"
    python ${DIR}/su_dawn.py \
        --mask_method "random" \
        --keep_ratio $keep_ratio \
        --epochs $epochs \
        --lr $lr \
        --save_file su_random_k${keep_ratio}_e${epochs}_l${lr}.tsv
fi