import numpy as np
import torch

# from datasets import load_dataset

import torch.nn.functional as F

import re

# ============lambada eval=============
class Evaluator_lambada:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'],padding='longest',truncation=True)
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids','attention_mask'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, hit = 0, 0
        for i, batch in enumerate(self.dataset):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            attention_mask = batch['attention_mask'].to(self.device).unsqueeze(0)
            label = input_ids[:,int(torch.sum(batch['attention_mask'])-1)]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, int(torch.sum(batch['attention_mask'])-2), :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if i % 200 == 0:
                print(i, ': ', hit / total)
        acc = hit / total
        return acc


# =====================================lambada end=====================


# =====================================piqa begin======================
class Evaluator_piqa:
    def __init__(self, dataset, tokenizer, device, _model_call):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self._model_call=_model_call

        def set_padding(examples):
            goal_len = max(len(elem) for elem in examples['goal'])
            choice1_len = max(len(elem) for elem in examples['sol1'])
            choice2_len = max(len(elem) for elem in examples['sol2'])
            self.padding_length = max(goal_len+choice1_len,goal_len+choice2_len)+20
            return None
        self.dataset.map(set_padding, batched=True)
        
        def tokenize_function(examples):
            out_doc = self._process_doc(examples)
            out_doc['context_enc'] = self.tokenizer(self.doc_to_text(out_doc),truncation=True)
            out_doc['continuation_enc1'] = self.tokenizer(self.doc_to_target1(out_doc),truncation=True)
            out_doc['continuation_enc2'] = self.tokenizer(self.doc_to_target2(out_doc),truncation=True)
            return out_doc
        
        self.dataset = self.dataset.map(tokenize_function, batched=False)
        # print(self.padding_length, self.dataset[0])

    def doc_to_text(self, doc):
        return "Question: " + doc["goal"] + "\nAnswer:"
    def doc_to_target1(self, doc):
        return " " + doc["choices"][0]
    def doc_to_target2(self, doc):
        return " " + doc["choices"][1]
    def _process_doc(self, doc):
        out_doc = {
            "goal": doc["goal"],
            "choices": [doc["sol1"], doc["sol2"]],
            "gold": doc["label"],
        }
        return out_doc
    def _loglikelihood_tokens(self, context_enc, continuation_enc, foward_func=None, sample=False):

        padding_length = self.padding_length
        inp = torch.tensor(
            (context_enc + continuation_enc)[:][:-1],
            dtype=torch.long,
        ).to(self.device)
        (inplen,) = inp.shape

        # since in _collate we make sure length is descending, the longest is always the first one.
        padding_length = (
            padding_length if padding_length is not None else inplen
        )

        # pad length from seq to padding_length
        inp = torch.cat(
            [
                inp,  # [seq]
                torch.zeros(padding_length - inplen, dtype=torch.long).to(
                    inp.device
                ),  # [padding_length - seq]
            ],
            dim=0,
        )
        inp = inp.unsqueeze(0)  # [1, padding_length]
        cont_toks = continuation_enc

        if sample:
            return inp
        if foward_func == None:
            output = self._model_call(inp)
            logits = F.log_softmax(
                output.logits, dim=-1
            ).cpu()  # [batch, padding_length, vocab]
        else:
            output = foward_func(inp)
            logits = F.log_softmax(
                output, dim=-1
            ).cpu()  # [batch, padding_length, vocab]

        # Slice to original seq length
        contlen = len(cont_toks)
        logits = logits[:,inplen - contlen : inplen]  # [1, seq, vocab]

        # Check if per-token argmax is exactly equal to continuation
        greedy_tokens = logits.argmax(dim=-1)
        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
            0
        )  # [1, seq]
        max_equal = (greedy_tokens == cont_toks).all()

        # Obtain log-probs at the corresponding continuation token indices
        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
            -1
        )  # [1, seq]

        # Answer: (log prob, is-exact-match)
        answer = (float(logits.sum()), bool(max_equal))

        return answer
    
    @torch.no_grad()
    def sample_batch(self):
        return torch.ones((1,self.padding_length),dtype=torch.int32)

    def my_collate_fn(self, batch):
        context_enc = batch['context_enc']['input_ids']
        continuation_enc1 = batch['continuation_enc1']['input_ids']
        input1 = self._loglikelihood_tokens(context_enc,continuation_enc1,sample=True)
        return input1.to(self.device)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for i, batch in enumerate(self.dataset):
            context_enc = batch['context_enc']['input_ids']
            continuation_enc1 = batch['continuation_enc1']['input_ids']
            continuation_enc2 = batch['continuation_enc2']['input_ids']
            label = batch['gold']
            outputs1 = self._loglikelihood_tokens(context_enc,continuation_enc1)[0]
            outputs2 = self._loglikelihood_tokens(context_enc,continuation_enc2)[0]
            pred = 0 if outputs1 > outputs2 else 1
            total += 1
            hit += pred == label
            if i % 200 == 0:
                print(i, ': ', hit / total)
        acc = hit / total
        return acc

# =====================================piqa end============================


# =====================================hellaswag begin=====================
class Evaluator_hellaswag:
    def __init__(self, dataset, tokenizer, device, _model_call):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self._model_call=_model_call
        
        def tokenize_function(examples):
            out_doc = self._process_doc(examples)
            out_doc['context_enc'] = self.tokenizer(self.doc_to_text(out_doc),truncation=True)
            out_doc['continuation_enc'] = self.tokenizer(self.doc_to_target(out_doc),truncation=True)
            return out_doc
        
        self.dataset = self.dataset.map(tokenize_function, batched=False)
        # self.dataset.set_format(type='torch', columns=['input_ids','attention_mask','context_enc','continuation_enc'])

        def set_padding(examples):
            context_enc = max(len(elem['input_ids']) for elem in examples['context_enc'])
            continuation_enc = max([max(map(len, ele['input_ids'])) for ele in examples['continuation_enc'] ])
            self.padding_length = context_enc+continuation_enc+20
            return None
        self.dataset.map(set_padding, batched=True)

    def _process_doc(self, doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": self.preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [" "+self.preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    def preprocess(self, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        # print("target1 ", " " + doc["choices"][0])
        return doc["choices"]

    def _loglikelihood_tokens(self, context_enc, continuation_enc, foward_func=None, sample=False):

        padding_length = self.padding_length
        inp = torch.tensor(
            (context_enc + continuation_enc)[:][:-1],
            dtype=torch.long,
        ).to(self.device)
        (inplen,) = inp.shape

        # since in _collate we make sure length is descending, the longest is always the first one.
        padding_length = (
            padding_length if padding_length is not None else inplen
        )

        # pad length from seq to padding_length
        inp = torch.cat(
            [
                inp,  # [seq]
                torch.zeros(padding_length - inplen, dtype=torch.long).to(
                    inp.device
                ),  # [padding_length - seq]
            ],
            dim=0,
        )
        inp = inp.unsqueeze(0)  # [1, padding_length]
        cont_toks = continuation_enc

        if sample:
            return inp
        if foward_func == None:
            output = self._model_call(inp)
            logits = F.log_softmax(
                output.logits, dim=-1
            ).cpu()  # [batch, padding_length, vocab]
        else:
            output = foward_func(inp)
            logits = F.log_softmax(
                output, dim=-1
            ).cpu()  # [batch, padding_length, vocab]

        # Slice to original seq length
        contlen = len(cont_toks)
        logits = logits[:,inplen - contlen : inplen]  # [1, seq, vocab]

        # Check if per-token argmax is exactly equal to continuation
        greedy_tokens = logits.argmax(dim=-1)
        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
            0
        )  # [1, seq]
        max_equal = (greedy_tokens == cont_toks).all()

        # Obtain log-probs at the corresponding continuation token indices
        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
            -1
        )  # [1, seq]

        # Answer: (log prob, is-exact-match)
        answer = (float(logits.sum()), bool(max_equal))

        return answer
    
    @torch.no_grad()
    def sample_batch(self):
        return torch.ones((1,self.padding_length),dtype=torch.int32)

    def my_collate_fn(self, batch):
        context_enc = batch['context_enc']['input_ids']
        continuation_enc1 = batch['continuation_enc']['input_ids'][0]
        input1 = self._loglikelihood_tokens(context_enc,continuation_enc1,sample=True)
        return input1.to(self.device)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for i, batch in enumerate(self.dataset):
            context_enc = batch['context_enc']['input_ids']
            label = batch['gold']
            outputs = []
            for continuation_enc in batch['continuation_enc']['input_ids']:
                output = self._loglikelihood_tokens(context_enc,continuation_enc)[0]
                outputs.append(output)
            pred = np.argmax(outputs)
            total += 1
            hit += pred == label
            if i % 200 == 0:
                print(i, ': ', hit / total)
        acc = hit / total
        return acc
# =====================================hellaswag end======================= 


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # ==========llama===================
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    # ==================================

    # from transformers import AutoTokenizer 
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)


    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    # from transformers import AutoTokenizer 
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    # ==========llama===================
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    # ==================================
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    # ==========llama===================
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    # ==================================

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
