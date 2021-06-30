echo "Enter the datasets you would like to perform HP search among, separated by spaces."
read datasets
echo "Performing hyperparameter search on $datasets"
learning_rates=(1e-5 2e-5 5e-5)
batch_sizes=(8 16)
mask_num_samples=1024
seed=42
method="label-square"

DIR=$(pwd)/transformers/examples/text-classification

for dataset in ${datasets[@]}; do
    for learning_rate in ${learning_rates[@]}; do
        for batch_size in ${batch_sizes[@]}; do
            file_name="bert-large-"$dataset"-"$learning_rate"-"$batch_size"-results.txt"
            experiment_name="hp_search, ${dataset}, lr=${learning_rate}, batch=${batch_size}"
            python ${DIR}/run_glue_sparse_update.py \
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

            acc=$(cat output/$dataset/eval_results_$dataset.txt)
            echo $acc
            echo -e $experiment_name", "$acc >> $file_name
        done
    done
done