export HF_DATASETS_CACHE="/fruitbasket/ylsung/.cache/huggingface/datasets"

for task in "cola" "mrpc" "qnli" "qqp"
do
    bash distributed_training_updated_2.sh $task 0 8 0.875 "1"
done
