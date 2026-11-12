"""Qwen2.5-7B-Instruct in 4-bit with Unsloth: load (optionally with a LoRA adapter), add LoRA, greedy generation."""
from unsloth import FastLanguageModel   # import before transformers / trl so Unsloth can patch them

import time

import torch

from common import build_prompt, fmt_time

MODEL_ID = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"     # standard bitsandbytes 4-bit (NF4)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model(path=None, max_seq_length=1024, for_training=False):
    """path: None = base model, or a saved LoRA adapter dir (Unsloth reads the base model from adapter_config.json)."""
    model, tok = FastLanguageModel.from_pretrained(model_name=str(path or MODEL_ID), max_seq_length=max_seq_length,
                                                   dtype=torch.float16, load_in_4bit=True)   # T4: fp16, no bf16
    if not for_training:
        FastLanguageModel.for_inference(model)
        tok.padding_side = "left"                     # batched generation needs left padding
    return model, tok


def add_lora(model, r=16, alpha=32, seed=42):
    """LoRA on all linear layers (choices justified in 08_sft_gold_7b_lora.ipynb)."""
    return FastLanguageModel.get_peft_model(model, r=r, lora_alpha=alpha, target_modules=LORA_TARGETS, lora_dropout=0,
                                            bias="none", use_gradient_checkpointing="unsloth", random_state=seed)


@torch.inference_mode()
def generate_greedy(model, tok, questions, batch_size=32, max_new_tokens=512, log=None):
    """Greedy answers (same settings as every earlier eval). Returns (texts, n_tokens)."""
    texts, n_tokens, t0 = [], [], time.time()
    for i in range(0, len(questions), batch_size):
        enc = tok([build_prompt(tok, q) for q in questions[i:i + batch_size]], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        new = out[:, enc.input_ids.shape[1]:]
        n_tokens += (new != tok.pad_token_id).sum(dim=1).tolist()
        texts += tok.batch_decode(new, skip_special_tokens=True)
        if log:
            el = time.time() - t0
            log.info(f"{len(texts)}/{len(questions)} answers | {len(texts) / el:.2f} answers/s | "
                     f"elapsed {fmt_time(el)} | ETA {fmt_time(el / len(texts) * (len(questions) - len(texts)))}")
    return texts, n_tokens
