export HF_DATASETS_CACHE="/fruitbasket/ylsung/.cache/huggingface/datasets"

for task in "stsb" "mrpc" "rte" "mnli"
do
    bash distributed_training_updated.sh $task 0
done
