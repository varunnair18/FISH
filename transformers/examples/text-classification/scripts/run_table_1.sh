DIR=$(pwd)/transformers/examples/text-classification

bash ${DIR}/scripts/run_glue.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 
bash ${DIR}/scripts/run_bitfit.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0
bash ${DIR}/scripts/run_random_baseline.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 
bash ${DIR}/scripts/run_glue_sparse_update.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 0.005 1024