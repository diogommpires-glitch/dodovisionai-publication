"""
DODO VISION AI v7 — QLoRA Fine-Tuning (Step 6)
Fine-tunes a local LLM on accumulated training pairs using Unsloth + SFTTrainer.
Produces LoRA adapter deployable via Ollama.

Usage:
    python train_qlora.py train --data calibration/llm_training_pairs.jsonl
    python train_qlora.py merge --adapter models/dodo_lora
    python train_qlora.py gguf --adapter models/dodo_lora
"""
import argparse, json, sys
from pathlib import Path

def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            e = json.loads(line.strip())
            data.append({
                "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are DODO Intelligence, a multimodal biometric surveillance AI analyzing heart rate, HRV, emotions, micro-expressions, gaze, vocal stress, and deception indicators.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{json.dumps(e.get('affect',{}), indent=2, default=str)}
{e.get('prompt','')}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{e.get('response','')}<|eot_id|>"""
            })
    return data

def train(args):
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing: {e}\npip install unsloth transformers trl datasets peft bitsandbytes")
        sys.exit(1)
    data = load_data(args.data)
    if len(data) < 10: print(f"Only {len(data)} pairs, need 10+"); sys.exit(1)
    print(f"Loaded {len(data)} pairs")
    ds = Dataset.from_list(data)
    model, tok = FastLanguageModel.from_pretrained(
        args.base_model, max_seq_length=2048, dtype=None, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=args.lora_r,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=args.lora_r, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth")
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds,
        args=TrainingArguments(output_dir=args.output, per_device_train_batch_size=2,
            gradient_accumulation_steps=4, warmup_steps=5, num_train_epochs=args.epochs,
            learning_rate=2e-4, fp16=True, logging_steps=1, save_strategy="epoch", optim="adamw_8bit"),
        dataset_text_field="text", max_seq_length=2048, packing=False)
    print("Training..."); trainer.train()
    model.save_pretrained(args.output); tok.save_pretrained(args.output)
    print(f"Saved to {args.output}")

def merge(args):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(args.adapter, max_seq_length=2048, dtype=None, load_in_4bit=True)
    model = model.merge_and_unload()
    out = args.output or args.adapter + "_merged"
    model.save_pretrained(out); tok.save_pretrained(out)
    Path(out, "Modelfile").write_text(f'FROM {out}\nSYSTEM "You are DODO Intelligence."\nPARAMETER temperature 0.3\n')
    print(f"Merged to {out}. Run: ollama create dodo -f {out}/Modelfile")

def gguf(args):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(args.adapter, max_seq_length=2048, dtype=None, load_in_4bit=True)
    out = args.output or args.adapter + "_gguf"
    model.save_pretrained_gguf(out, tok, quantization_method="q4_k_m")
    print(f"GGUF saved to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--output", default="models/dodo_lora")
    t.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    t.add_argument("--epochs", type=int, default=3); t.add_argument("--lora-r", type=int, default=16)
    m = sub.add_parser("merge"); m.add_argument("--adapter", required=True); m.add_argument("--output")
    g = sub.add_parser("gguf"); g.add_argument("--adapter", required=True); g.add_argument("--output")
    args = p.parse_args()
    {"train": train, "merge": merge, "gguf": gguf}.get(args.cmd, lambda a: p.print_help())(args)
