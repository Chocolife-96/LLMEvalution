import torch
import torch.nn as nn

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto', local_files_only=True)
    # from transformers import LlamaForCausalLM, AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained("/mnt/glusterfs-pvc/usr/dayou.du/models/hf-llama-2-7b")
    model.seqlen = 2048
    return model

if __name__ == '__main__':
    from evaluation_ppl import *
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'ckpt', type=str,
        help='model ckpt'
    )
    args = parser.parse_args()

    model = get_llama(args.model)
    # =========load your ckpt========
    checkpoint = torch.load(args.mckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    # ===============================
    model.eval()
    # =========tokenizer prepare=====
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # =========datasets==============
    datasets = ['wikitext2', 'ptb', 'c4']

    evaluation_ppl(model, tokenizer, datasets)
