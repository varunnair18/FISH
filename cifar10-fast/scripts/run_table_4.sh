num_samples=1024
epochs=100
lr=0.4

DIR=cifar10-fast

# sparse update for the FISH mask
for keep_ratio in 0.005 0.02 0.1
do
    for lr in 0.4 0.2 0.08 0.04 0.02
    do
        # Allow the FISH mask to be updated every epoch
        bash ${DIR}/scripts/small_checkpoints_fish.sh ${num_samples} ${keep_ratio} ${epochs} ${lr} 0
        # Fix the FISH mask all the time
        bash ${DIR}/scripts/small_checkpoints_fish.sh ${num_samples} ${keep_ratio} ${epochs} ${lr} 1

        # Allow the random mask to be updated every epoch
        bash ${DIR}/scripts/small_checkpoints_random.sh ${keep_ratio} ${epochs} ${lr} 0
        # Fix the random mask all the time
        bash ${DIR}/scripts/small_checkpoints_random.sh ${keep_ratio} ${epochs} ${lr} 1
    done
done
