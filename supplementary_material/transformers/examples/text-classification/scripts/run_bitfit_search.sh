datasets=$1
seed=$2

source $(pwd)/lib.sh

# methods=("expect-square")
method="bias"
sample_size=1024

bias_learning_rates=(1e-3 5e-4 1e-4)

for dataset in ${datasets[@]}; do
    for learning_rate in ${bias_learning_rates[@]}; do
        index="$(($(get_dataset_index $dataset) - 0))"

        output_dir="output/$dataset/"

        batch_size=${batch_sizes[$index]}

        file_name="bert-large-"$dataset"-bitfit-"$method"-"$learning_rate"-"$batch_size"-"$seed"-results.txt"
        experiment_name="bitfit, ${dataset}, method=${method}, lr=${learning_rate}, batch=${batch_size}, seed=${seed}"
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
            --num_samples $sample_size \
            --mask_method $method \
            --save_steps 0 \
            --seed $seed \
            --overwrite_output_dir

        acc=$(cat ${output_dir}eval_results_$dataset.txt)
        echo $acc
        echo -e $experiment_name", "$acc >> $file_name

        # Save history file using same filename
        len="$((${#file_name}-3))"

        mv "${output_dir}eval_history_${dataset}.json" "${file_name::$len}json"

        # Save test set results
        mv "${output_dir}test_results_${dataset}.txt" "test-${file_name}"

        mv "${output_dir}test_results_${dataset}.tsv" "test-${file_name::$len}tsv"

        if [[ $dataset == "mnli" ]]; then
            $dataset="mnli-mm"

           file_name="bert-large-"$dataset"-bitfit-"$method"-"$learning_rate"-"$batch_size"-"$seed"-results.txt"
            experiment_name="bitfit, ${dataset}, method=${method}, lr=${learning_rate}, batch=${batch_size}, seed=${seed}"
            
            acc=$(cat ${output_dir}eval_results_$dataset.txt)
            echo $acc
            echo -e $experiment_name", "$acc >> $file_name

            # Save mnli-mm test results
            mv "${output_dir}test_results_${dataset}.txt" "test-${file_name}"

            mv "${output_dir}test_results_${dataset}.tsv" "test-${file_name::$len}tsv"
        fi

    done
done