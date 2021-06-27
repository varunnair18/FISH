datasets=$1
seed=$2

source $(pwd)/lib.sh

mkdir -p tsv

mask_num_samples=1024
method="random"

for dataset in ${datasets[@]}; do
    index="$(($(get_dataset_index $dataset) - 0))"

    output_dir="output/$dataset/"

    learning_rate=${random_baseline_learning_rates[$index]}
    batch_size=${batch_sizes[$index]}

    file_name="bert-large-"$dataset"-"$learning_rate"-"$batch_size"-"$seed"-random-results"
    experiment_name="random_baseline, ${dataset}, lr=${learning_rate}, batch=${batch_size}, seed=${seed}"
    python run_glue_sparse_update.py \
        --model_name_or_path bert-large-cased-whole-word-masking \
        --task_name $dataset \
        --do_train \
        --do_eval \
        --do_predict \
        --max_seq_length 128 \
        --per_device_train_batch_size $batch_size \
        --learning_rate $learning_rate \
        --num_train_epochs 7 \
        --output_dir output/$dataset/ \
        --evaluation_strategy "epoch" \
        --num_samples $mask_num_samples \
        --keep_ratio 0.005 \
        --mask_method $method \
        --exp_name $file_name \
        --save_steps 0 \
        --seed $seed \
        --overwrite_output_dir

    # Save evaluation set results
    acc=$(cat ${output_dir}eval_results_${file_name}.txt)
    echo $acc
    echo -e $experiment_name", "$acc >> "${output_dir}eval_results_${file_name}.txt"

    mv "${output_dir}eval_results_${file_name}.txt" "output_logs/${file_name}.txt"

    # Save history file using same filename
    mv "${output_dir}eval_history_${file_name}.json" "output_logs/history_${file_name}.json"

    # Save test set results
    mv "${output_dir}test_results_${file_name}.txt" "output_logs/test-${file_name}.txt"

    mv "${output_dir}test_results_${file_name}.tsv" "tsv/test-${file_name}.tsv"

    if [[ $dataset == "mnli" ]]; then
        dataset="mnli-mm"

        file_name="bert-large-"$dataset"-"$learning_rate"-"$batch_size"-"$seed"-random-results"
        experiment_name="random_baseline, ${dataset}, lr=${learning_rate}, batch=${batch_size}, seed=${seed}"
        
        acc=$(cat ${output_dir}eval_results_${file_name}.txt)
        echo $acc
        echo -e $experiment_name", "$acc >> "${output_dir}eval_results_${file_name}.txt"

        mv "${output_dir}eval_results_${file_name}.txt" "output_logs/${file_name}.txt"

        # Save test set results
        mv "${output_dir}test_results_${file_name}.txt" "output_logs/test-${file_name}.txt"

        mv "${output_dir}test_results_${file_name}.tsv" "tsv/test-${file_name}.tsv"
    fi
done