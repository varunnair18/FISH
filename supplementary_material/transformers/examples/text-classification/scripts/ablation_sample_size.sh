datasets=$1
seed=$2

source $(pwd)/lib.sh

sample_sizes=(32 256 1024 16384)
method="label-square"

for dataset in ${datasets[@]}; do
    index="$(($(get_dataset_index $dataset) - 0))"

    output_dir="output/$dataset/"

    learning_rate=${learning_rates[$index]}
    batch_size=${batch_sizes[$index]}

    for sample_size in ${sample_sizes[@]}; do
        file_name="bert-large-"$dataset"-ablation-sample-size-"$sample_size"-"$learning_rate"-"$batch_size"-"$seed"-results.txt"
        experiment_name="ablation-sample-size, ${dataset}, sample-size=${sample_size}, lr=${learning_rate}, batch=${batch_size}, seed=$seed"
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
            --num_samples $sample_size \
            --keep_ratio 0.005 \
            --mask_method $method \
            --exp_name $file_name \ 
            --save_steps 0 \
            --seed $seed \
            --overwrite_output_dir

        acc=$(cat ${output_dir}eval_results_$dataset.txt)
        echo $acc
        echo -e $experiment_name", "$acc >> $file_name

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

            file_name="bert-large-"$dataset"-"$learning_rate"-"$batch_size"-"$seed"-"$keep_ratio"-sparse-results"
            experiment_name="sparse_experiment, ${dataset}, lr=${learning_rate}, batch=${batch_size}, seed=${seed}, ratio=${keep_ratio}"
            
            acc=$(cat ${output_dir}eval_results_${file_name}.txt)
            echo $acc
            echo -e $experiment_name", "$acc >> "${output_dir}eval_results_${file_name}.txt"

            mv "${output_dir}eval_results_${file_name}.txt" "output_logs/${file_name}.txt"

            # Save test set results
            mv "${output_dir}test_results_${file_name}.txt" "output_logs/test-${file_name}.txt"

            mv "${output_dir}test_results_${file_name}.tsv" "tsv/test-${file_name}.tsv"
        fi
    done
done