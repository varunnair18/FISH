export HF_DATASETS_CACHE="/fruitbasket/ylsung/.cache/huggingface/datasets"

for task in "stsb" "rte" "sst2" "mnli"
do
    bash distributed_training_updated_2.sh $task 0 8 0.875 "0"
done
