num_samples=$1
keep_ratio=$2
epochs=$3
merge_steps=$4
lr=$5
split=$6

DIR=cifar10-fast/

python ${DIR}/asgd_dawn.py \
        --num_samples $num_samples \
        --keep_ratio $keep_ratio \
        --epochs $epochs \
        --merge_steps $merge_steps \
        --lr $lr \
        --split $split \
        --warmup_epoch 5 \
        --same_mask \
        --save_file logs/asgd_m${merge_steps}_k${keep_ratio}_e${epochs}_l${lr}_s${split}_w5.tsv