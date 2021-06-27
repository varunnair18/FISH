TASK_NAME=$1
SEED=$2

source $(pwd)/dt_lib.sh

full_train_lr=1e-5
method="label-square"
keep_ratio=0.005
mask_num_samples=256

index="$(($(get_dataset_index $TASK_NAME) - 0))"

lr=${learning_rates[$index]}
BS=${batch_sizes[$index]}

EPOCH=7
HEPOCH=3.5

EVAL_INTERVAL=${eval_intervals[$index]}
GPU="0"

# Check the pretrained weight is existed, or create one
pretrained_name=${TASK_NAME}_pretrained@${SEED}
find=$(find tmp -name ${pretrained_name})

if [ ${#find} -ge 1 ]
then 
    echo $find
    echo "Find pretrained weight"
else
    echo "Not find pretrained weight"

    # extract and fix the pretrained_weights
    CUDA_VISIBLE_DEVICES=$GPU python extract_untrained_pretrained_weight.py \
    --model_name_or_path bert-large-cased-whole-word-masking \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate $lr \
    --num_train_epochs 7 \
    --output_dir ./tmp/${pretrained_name} \
    --evaluation_strategy "epoch" \
    --save_steps 10000 \
    --overwrite_output_dir \
    --seed ${SEED}
fi

# Train first baseline: fully updated model trained for double epoch
CUDA_VISIBLE_DEVICES=$GPU python run_glue_iter.py \
  --model_name_or_path ./tmp/${pretrained_name} \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $BS \
  --learning_rate $full_train_lr \
  --num_train_epochs $EPOCH \
  --output_dir ./tmp/${TASK_NAME}_fully_normal@${SEED}/ \
  --evaluation_strategy "epoch" \
  --num_samples $mask_num_samples \
  --keep_ratio $keep_ratio \
  --mask_method $method \
  --save_steps 0 \
  --split 1 \
  --overwrite_output_dir \
  --seed ${SEED} \
  --same_classifier \
  --diff_aggr_method "sum" \
  --evaluate_interval $EVAL_INTERVAL 


for MERGE_STEP in 10 30 100
do

# Train ASGD of our approach
CUDA_VISIBLE_DEVICES=$GPU python run_glue_asgd.py \
  --model_name_or_path ./tmp/${pretrained_name} \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $BS \
  --learning_rate $lr \
  --num_train_epochs $HEPOCH \
  --output_dir ./tmp/${TASK_NAME}_sparse_asgd_merge${MERGE_STEP}@${SEED}/ \
  --evaluation_strategy "epoch" \
  --num_samples $mask_num_samples \
  --keep_ratio $keep_ratio \
  --mask_method $method \
  --save_steps 0 \
  --split 2 \
  --overwrite_output_dir \
  --seed ${SEED} \
  --same_classifier \
  --diff_aggr_method "sum" \
  --evaluate_interval $EVAL_INTERVAL \
  --merge_step $MERGE_STEP


# Train second baseline: fully updated asgd model by summing the difference
CUDA_VISIBLE_DEVICES=$GPU python run_glue_asgd.py \
  --model_name_or_path ./tmp/${pretrained_name} \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $BS \
  --learning_rate $full_train_lr \
  --num_train_epochs $HEPOCH \
  --output_dir ./tmp/${TASK_NAME}_fully_asgd_sum_merge${MERGE_STEP}@${SEED}/ \
  --evaluation_strategy "epoch" \
  --num_samples $mask_num_samples \
  --keep_ratio $keep_ratio \
  --mask_method "all_ones" \
  --save_steps 0 \
  --split 2 \
  --overwrite_output_dir \
  --seed ${SEED} \
  --same_classifier \
  --diff_aggr_method "sum" \
  --evaluate_interval $EVAL_INTERVAL \
  --merge_step $MERGE_STEP


CUDA_VISIBLE_DEVICES=$GPU python run_glue_asgd.py \
  --model_name_or_path ./tmp/${pretrained_name} \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $BS \
  --learning_rate $full_train_lr \
  --num_train_epochs $HEPOCH \
  --output_dir ./tmp/${TASK_NAME}_fully_asgd_mean_merge${MERGE_STEP}@${SEED}/ \
  --evaluation_strategy "epoch" \
  --num_samples $mask_num_samples \
  --keep_ratio $keep_ratio \
  --mask_method "all_ones" \
  --save_steps 0 \
  --split 2 \
  --overwrite_output_dir \
  --seed ${SEED} \
  --same_classifier \
  --diff_aggr_method "mean" \
  --evaluate_interval $EVAL_INTERVAL \
  --merge_step $MERGE_STEP

done