export HF_DATASETS_CACHE="/fruitbasket/ylsung/.cache/huggingface/datasets"

for task in "cola" "mrpc" "qnli" "qqp"
do
    bash distributed_training_updated.sh $task 0
done
