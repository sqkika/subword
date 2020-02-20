#! /bin/bash
set -x #echo on

data_vocab_path=$1
save_path=$2
graph_save_path=$3
config=$4
if [ "x$5" != "x" ]; then
   gpu=$5
   else
   gpu=""
fi
echo "Using GPU $gpu"

CUDA_VISIBLE_DEVICES=$gpu python3 main.py --save_path=${save_path} --graph_save_path=${graph_save_path} --data_path=${data_vocab_path} --vocab_path=${data_vocab_path} --model_config=${config}
