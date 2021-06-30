datasets=$1
seed=$2

source $(pwd)/scripts/lib.sh

# methods=("expect-square")
methods=("random" "bias")
sample_size=1024

for dataset in ${datasets[@]}; do
    for method in ${methods[@]}; do
        index="$(($(get_dataset_index $dataset) - 0))"

        output_dir="output/$dataset/"

        learning_rate=${learning_rates[$index]}
        batch_size=${batch_sizes[$index]}

        file_name="bert-large-"$dataset"-ablation-gradient-type-"$method"-"$learning_rate"-"$batch_size"-"$seed"-results.txt"
        experiment_name="ablation-gradient-type, ${dataset}, method=${method}, lr=${learning_rate}, batch=${batch_size}, seed=${seed}"
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
            --keep_ratio 0.005 \
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
    done
done