echo "imported dt_lib."
datasets_available=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb")
learning_rates=(5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5)
random_baseline_learning_rates=(5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5)
bitfit_learning_rates=(1e-3 1e-3 5e-4 1e-4 5e-4 5e-4 5e-4 5e-4)
batch_sizes=(16 16 16 16 16 8 16 16)
eval_intervals=(1 50 1 15 50 1 15 1)

function get_dataset_index() {
    i=-1

    # Get index of experiment
    for j in "${!datasets_available[@]}"; do
        if [[ "${datasets_available[$j]}" = "$1" ]]; then
            i=$j
        fi
    done
    echo $i
}

function get_lr() {
    echo $learning_rates[$1]
}

function get_batch_size() {
    echo $batch_sizes[$1]
}

