num_samples=$1
keep_ratio=$2
epochs=$3
lr=$4
fix_mask=$5

DIR=cifar10-fast

if [ $fix_mask == 1 ]; then
    echo "Fix the mask during training"
    python ${DIR}/su_dawn.py \
        --num_samples $num_samples \
        --keep_ratio $keep_ratio \
        --epochs $epochs \
        --lr $lr \
        --fix_mask \
        --save_file su_s${num_samples}_k${keep_ratio}_e${epochs}_l${lr}_fix.tsv
else
    echo "The mask is allowed to be updated every epoch"
    python ${DIR}/su_dawn.py \
        --num_samples $num_samples \
        --keep_ratio $keep_ratio \
        --epochs $epochs \
        --lr $lr \
        --save_file su_s${num_samples}_k${keep_ratio}_e${epochs}_l${lr}.tsv
fi

