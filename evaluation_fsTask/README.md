## Download Data

Recommend to run ```bash download.sh``` first

## Usage - Example
### Install
https://github.com/EleutherAI/lm-evaluation-harness
### QA Task
```
python lmeval.py \
    --model hf-causal-experimental \
    --model_args pretrained=model_or_path,use_accelerate=True \
    --num_fewshot=0 \
    --batch_size=2 \
    --tasks piqa,hellaswag,lambada_openai 
```
### MMLU
```
python lmeval.py \
    --model hf-causal-experimental \
    --model_args pretrained=model_or_path,use_accelerate=True \
    --num_fewshot=5 \
    --batch_size=2 \
    --tasks hendrycksTest-* \
    --utput_path (ckpts/ckpt_nf4_chatQAT/checkpoint-100/mmlu_nf4.json)

python mmlu_avg.py --json_path ckpts/ckpt_nf4_chatQAT/checkpoint-100/mmlu_nf4.json
```



