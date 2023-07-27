#!/bin/bash

# 默认参数值
model_or_path=""
task="piqa,hendrycksTest-*,lambada_openai,hellaswag"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --model_or_path)
        model_or_path="$2"
        shift # 这里需要多移动一个参数，因为我们的参数是键值对的形式
        shift
        ;;
        --task)
        task="$2"
        shift
        shift
        ;;
        *) # 如果参数无法识别，则忽略
        shift
        ;;
    esac
done

# 打印参数信息（可选步骤，用于检查参数是否正确）
echo "model_or_path: $model_or_path"
echo "task: $task"
# 调用 Python 脚本
/root/model/miniconda3/envs/purellm/bin/python3.10 lmeval.py --model hf-causal-experimental --model_args "pretrained=$model_or_path,use_accelerate=True" --num_fewshot=0 --batch_size=2 --tasks="$task"
