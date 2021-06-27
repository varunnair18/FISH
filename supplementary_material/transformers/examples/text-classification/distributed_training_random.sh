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

for MERGE_STEP in 10 30 100
do

# Train second baseline: fully updated asgd model by summing the difference
python run_glue_asgd.py \
  --model_name_or_path bert-large-cased-whole-word-masking \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $BS \
  --learning_rate $lr \
  --num_train_epochs $HEPOCH \
  --output_dir ./tmp/${TASK_NAME}_random_asgd_sum_merge${MERGE_STEP}@${SEED}/ \
  --evaluation_strategy "epoch" \
  --num_samples $mask_num_samples \
  --keep_ratio $keep_ratio \
  --mask_method "random" \
  --save_steps 0 \
  --split 2 \
  --overwrite_output_dir \
  --seed ${SEED} \
  --same_classifier \
  --diff_aggr_method "sum" \
  --evaluate_interval $EVAL_INTERVAL \
  --merge_step $MERGE_STEP
done