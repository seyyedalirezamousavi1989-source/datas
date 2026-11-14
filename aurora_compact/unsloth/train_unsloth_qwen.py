"""Unsloth LoRA fine-tune template for the AURORA compact dataset.  NOT executed in this project (no training has been run).

Check before running: (1) the Hugging Face repo that matches your Ollama tag `qwen3.5:4b` (set --model), (2) your Unsloth/TRL versions
(argument names move between releases), (3) that the chat markers below match the model's chat template (ChatML: <|im_start|>).
    pip install unsloth trl datasets
    python train_unsloth_qwen.py --model <hf-repo-of-qwen3.5-4b> --epochs 2
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="HF repo / local path of the base model (not guessed here)")
ap.add_argument("--data", default=".")
ap.add_argument("--prefix", default="aurora_compact_v2", help="aurora_compact (v1) | aurora_compact_v2 | aurora_compact_v2_tools")
ap.add_argument("--tools_json", default=None, help="forex_tools.json; passed to the chat template for rows that contain tool_calls")
ap.add_argument("--out", default="aurora_lora")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max_len", type=int, default=4096)
a = ap.parse_args()

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_len, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], use_gradient_checkpointing="unsloth")

ds = load_dataset("json", data_files={"train": f"{a.data}/{a.prefix}_train.jsonl", "val": f"{a.data}/{a.prefix}_val.jsonl"})


import json
TOOLS = json.load(open(a.tools_json, encoding="utf-8")) if a.tools_json else None


def fmt(batch):
    # enable_thinking=False keeps the training text free of <think> blocks (answers are direct); drop the kwarg if your template lacks it.
    # Tool schemas are given only to rows that contain a tool call (keeps ordinary rows short).
    out = []
    for m in batch["messages"]:
        has_call = any(x.get("tool_calls") for x in m)
        out.append(tok.apply_chat_template(m, tools=TOOLS if (TOOLS and has_call) else None, tokenize=False, add_generation_prompt=False, enable_thinking=False))
    return {"text": out}


ds = ds.map(fmt, batched=True, remove_columns=ds["train"].column_names)
trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds["train"], eval_dataset=ds["val"],
                     args=SFTConfig(dataset_text_field="text", max_seq_length=a.max_len, per_device_train_batch_size=2, gradient_accumulation_steps=8,
                                    num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=10,
                                    eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=200, bf16=True, optim="adamw_8bit",
                                    output_dir=a.out, report_to="none", seed=2026))
# loss only on the assistant turn (system prompt + data context are not trained on)
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
trainer.train()
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
# optional: model.save_pretrained_gguf(a.out + "_gguf", tok, quantization_method="q4_k_m")  -> import into Ollama with a Modelfile
