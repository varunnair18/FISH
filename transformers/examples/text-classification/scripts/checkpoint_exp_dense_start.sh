checkpoint_freq=$1
echo "Performing checkpointing every $checkpoint_freq epochs"
echo "Enter the datasets you would like to perform checkpointing experiments among, separated by spaces."
read datasets

datasets_available=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb")
learning_rates=(5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5)
batch_sizes=(16 16 16 16 16 8 16 16)
mask_num_samples=1024
method="label-square"
seed=0
# checkpoint_freq=0

for dataset in ${datasets[@]}; do
    index=-1

    # Get index of experiment
    for j in "${!datasets_available[@]}"; do
        if [[ "${datasets_available[$j]}" = "${dataset}" ]]; then
            echo "${j}"
            index=$j
        fi
    done

    learning_rate=${learning_rates[$index]}
    batch_size=${batch_sizes[$index]}

    python run_glue_sparse_update.py \
        --model_name_or_path bert-large-cased-whole-word-masking \
        --task_name $dataset \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size $batch_size \
        --learning_rate $learning_rate \
        --num_train_epochs 7 \
        --output_dir output/$dataset/ \
        --evaluation_strategy "epoch" \
        --num_samples $mask_num_samples \
        --keep_ratio 0.005 \
        --mask_method $method \
        --save_steps 0 \
        --seed $seed \
        --overwrite_output_dir


    file_name="bert-large-"$dataset"-"$learning_rate"-"$batch_size"-"$checkpoint_freq"-"$seed"-checkpointing-results.txt"
    experiment_name="checkpointing_exp, ${dataset}, lr=${learning_rate}, batch=${batch_size}, seed=${seed}, checkpoint_freq=${checkpoint_freq}"
    python run_glue_checkpointing.py \
        --model_name_or_path output/$dataset \
        --task_name $dataset \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size $batch_size \
        --learning_rate $learning_rate \
        --num_train_epochs 7 \
        --output_dir output/$dataset/ \
        --evaluation_strategy "epoch" \
        --num_samples $mask_num_samples \
        --keep_ratio 0.005 \
        --mask_method $method \
        --save_steps 0 \
        --seed $seed \
        --checkpoint_frequency $checkpoint_freq \
        --overwrite_output_dir

    acc=$(cat output/$dataset/eval_results_$dataset.txt)
    echo $acc
    acc_mm=""

    if [[ $dataset == "mnli" ]]; then
        acc_mm=$(cat output/$dataset/eval_results_${dataset}-mm.txt)
    fi

    echo -e $experiment_name", "$acc", "$acc_mm >> $file_name
done

# mask_num_samples=1024
# method="label-square"

# for dataset in ${datasets[@]}; do
#     for learning_rate in ${learning_rates[@]}; do
#         for batch_size in ${batch_sizes[@]}; do
#             file_name="bert-large-"$dataset"-"$learning_rate"-"$batch_size"-results.txt"
#             experiment_name="hp_search, ${dataset}, lr=${learning_rate}, batch=${batch_size}"
#             python run_glue_sparse_update.py \
#               --model_name_or_path bert-large-cased-whole-word-masking \
#               --task_name $dataset \
#               --do_train \
#               --do_eval \
#               --max_seq_length 128 \
#               --per_device_train_batch_size $batch_size \
#               --learning_rate $learning_rate \
#               --num_train_epochs 7 \
#               --output_dir output/$dataset/ \
#               --evaluation_strategy "epoch" \
#               --num_samples $mask_num_samples \
#               --keep_ratio 0.005 \
#               --mask_method $method \
#               --save_steps 0 \
#               --overwrite_output_dir

#             acc=$(cat output/$dataset/eval_results_$dataset.txt)
#             echo $acc
#             echo -e $experiment_name", "$acc >> $file_name
#         done
#     done
# done