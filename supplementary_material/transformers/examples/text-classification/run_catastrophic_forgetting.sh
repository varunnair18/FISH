datasets=$1
seed=$2

source $(pwd)/lib.sh


for dataset in ${datasets[@]}; do
    
    index="$(($(get_dataset_index $dataset) - 0))"

    learning_rate=1e-5
    batch_size=16

    output_dir="output/$dataset/"

    file_name="bert-large-"$dataset"-"$learning_rate"-"$batch_size"-"$seed"-normal-results.txt"
    experiment_name="normal_experiment, ${dataset}, lr=${learning_rate}, batch=${batch_size}, seed=${seed}"
    python run_glue.py \
        --model_name_or_path bert-large-cased-whole-word-masking \
        --task_name $dataset \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size $batch_size \
        --learning_rate $learning_rate \
        --num_train_epochs 7 \
        --output_dir $output_dir \
        --evaluation_strategy "epoch" \
        --save_steps 0 \
        --seed $seed \
        --overwrite_output_dir

    acc=$(cat ${output_dir}eval_results_$dataset.txt)
    echo $acc
    echo -e $experiment_name", "$acc >> $file_name

    # Save history file using same filename
    len="$((${#file_name}-3))"
    mv "${output_dir}eval_history_${dataset}.json" "${file_name::$len}json"
done