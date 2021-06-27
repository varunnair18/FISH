TASK_NAME=rte
file_name="bert-large-"$TASK_NAME"-results.txt" # save experiment results

lr=5e-5

samples=2490
experiment_name="["$method"] (bs=8, lr="$lr"), samples="$samples
echo $experiment_name

method="expect-square"
python run_glue_sparse_update.py \
    --model_name_or_path bert-large-cased-whole-word-masking \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate $lr \
    --num_train_epochs 7 \
    --output_dir ./tmp/$TASK_NAME \
    --evaluation_strategy "epoch" \
    --mask_method $method \
    --num_samples $samples \
    --keep_ratio 0.005 \
    --save_steps 10000 \
    --overwrite_output_dir

# extract accuracy
acc=$(cat ./tmp/$TASK_NAME/eval_results_$TASK_NAME.txt)
echo $acc
echo -e $experiment_name", "$acc >> $file_name

# python run_glue_sparse_update.py \
#     --model_name_or_path bert-large-cased-whole-word-masking \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_eval \
#     --max_seq_length 128 \
#     --per_device_train_batch_size 8 \
#     --learning_rate $lr \
#     --num_train_epochs 7 \
#     --output_dir ./tmp/$TASK_NAME"_random_baseline"/ \
#     --evaluation_strategy "epoch" \
#     --mask_method "random" \
#     --keep_ratio 0.005 \
#     --save_steps 100 \
#     --overwrite_output_dir

# for epoch in 1 2 3
# do
#   python run_glue_last_layer.py \
#     --model_name_or_path bert-large-cased-whole-word-masking \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_eval \
#     --max_seq_length 128 \
#     --per_device_train_batch_size 16 \
#     --learning_rate $lr \
#     --num_train_epochs $epoch \
#     --output_dir ./tmp/$TASK_NAME"_normal_train_epoch"$epoch"_reset"/ \
#     --evaluation_strategy "epoch" \
#     --save_steps 100 \
#     --overwrite_output_dir

#   method="expect-square"
#   python run_glue_sparse_update.py \
#     --model_name_or_path bert-large-cased-whole-word-masking \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_eval \
#     --max_seq_length 128 \
#     --per_device_train_batch_size 16 \
#     --learning_rate $lr \
#     --num_train_epochs 7 \
#     --output_dir ./tmp/$TASK_NAME"_es_load_epoch"$epoch"_reset"/ \
#     --evaluation_strategy "epoch" \
#     --mask_method $method \
#     --num_samples $samples \
#     --keep_ratio 0.005 \
#     --save_steps 100 \
#     --overwrite_output_dir \
#     --model_name_or_path ./tmp/$TASK_NAME"_normal_train_epoch"$epoch"_reset"/ \
#     --model_load_path_second ./tmp/$TASK_NAME"_random_classifier"/

#   for ((i=100;i<=1000;i=i+100))
#   do
#       python generate_masks.py \
#       --model_name_or_path bert-large-cased-whole-word-masking \
#       --task_name $TASK_NAME \
#       --do_train \
#       --do_eval \
#       --max_seq_length 128 \
#       --per_device_train_batch_size 16 \
#       --learning_rate $lr \
#       --num_train_epochs 7 \
#       --output_dir ./tmp/$TASK_NAME"_es_load_epoch"$epoch"_reset"/ \
#       --model_name_or_path ./tmp/$TASK_NAME"_es_load_epoch"$epoch"_reset"/checkpoint-$i \
#       --evaluation_strategy "epoch" \
#       --num_samples $samples \
#       --keep_ratio 0.005 \
#       --mask_method "expect-square" \
#       --save_steps 100 
#   done
#   python results_analysis.py mask_similarity_checkpoints ./tmp/$TASK_NAME"_es_load_epoch"$epoch"_reset" "img/mask-expect-square-load-epoch"$epoch"-reset.png"
# done
