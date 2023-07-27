## Download Data

Recommend to run ```bash download.sh``` first

## Usage - Example
### Install
https://github.com/EleutherAI/lm-evaluation-harness
### QA Task
```
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=model_or_path,use_accelerate=True \
    --num_fewshot=0 \
    --batch_size=2 \
    --tasks piqa,hellaswag,lambada_openai 
```
### MMLU
```
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=model_or_path,use_accelerate=True \
    --num_fewshot=5 \
    --batch_size=2 \
    --tasks hendrycksTest-*
```


