epochs=$1
merge_steps=$2
lr=$3
split=$4

DIR=cifar10-fast/

python ${DIR}/asgd_dawn.py \
    --epochs $epochs \
    --merge_steps $merge_steps \
    --mask_method "all_ones" \
    --lr $lr \
    --split $split \
    --save_file logs/asgd_m${merge_steps}_k1.0_e${epochs}_l${lr}_s${split}.tsv