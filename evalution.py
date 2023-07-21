import torch
import torch.nn as nn
from datasets import load_dataset

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, cache_dir='/gptq/hub', torch_dtype='auto')
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        if i % 10 == 0:
            print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )

    args = parser.parse_args()

    model = get_llama(args.model)
    model.eval()

    
    datasets = ['wikitext2', 'ptb', 'c4'] 
    # datasets = ['wikitext2']
    if args.new_eval:
      datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)


    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = "[PAD]"
    # ============lambada==================
    dataset = load_dataset('lambada', cache_dir='/gptq/datasets', split='validation')
    evaluator_lambada = Evaluator_lambada(dataset, tokenizer, 'cuda')
    acc_lambada = evaluator_lambada.evaluate(model.cuda())
    evaluator_lambada = None
    dataset = None
    print("lambada: ", acc_lambada)
    torch.cuda.empty_cache()
    # # =============piqa=====================
    dataset = load_dataset('piqa', split='validation')
    evaluator_piqa = Evaluator_piqa(dataset, tokenizer, 'cuda', model)
    acc_piqa = evaluator_piqa.evaluate(model.cuda())
    evaluator_piqa = None
    dataset = None
    print("piqa: ", acc_piqa)
    torch.cuda.empty_cache()
    # # =============hellaswag================
    dataset = load_dataset('hellaswag', split='validation')
    evaluator_hellaswag = Evaluator_hellaswag(dataset, tokenizer, 'cuda', model)
    acc_hellaswag = evaluator_hellaswag.evaluate(model.cuda())
    evaluator_hellaswag = None
    dataset = None
    print("hellaswag: ", acc_hellaswag)
    torch.cuda.empty_cache()