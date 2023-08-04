from lm_eval import evaluator, tasks
from utils import LMEvalAdaptor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)

model = AutoModelForCausalLM(args.model_path)
tokenizer = AutoTokenizer(args.model_path)

lm_eval_model = LMEvalAdaptor(args.model_path, model, tokenizer, 2)

results = evaluator.simple_evaluate(
    model=lm_eval_model,
    tasks=task_names,
    batch_size=2,
    no_cache=True,
    num_fewshot=args.num_fewshot
)