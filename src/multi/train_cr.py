"""
ETAPE 4 — Prompt-based FINE-TUNING (manuel) sur CR (fichier NEUF, isole).

Meme protocole que pour SST-2 : 5 seeds x grille (lr, batch), selection sur val
(departage par perte val), moyenne +/- ecart-type. Cible papier (Table 3, CR) : 87.0.
Sauvegarde 04_prompt_ft_manual.json / .txt.
"""

import json
import os
import statistics
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data_cr import load_cr_fewshot

transformers.logging.set_verbosity_error()

MODEL_NAME = "roberta-large"
TEMPLATE = "{sentence} It was {mask} ."
LABEL_WORDS = {1: "great", 0: "terrible"}
SEEDS = [13, 21, 42, 87, 100]
GRID = [(1e-5, 2), (1e-5, 4), (2e-5, 4), (2e-5, 8)]
EPOCHS = 60
PAPER_TARGET = 90.3
OUT_DIR = "outputs_cr"


def get_label_token_ids(tokenizer):
    return {cls: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + w)[0])
            for cls, w in LABEL_WORDS.items()}


def forward_logits(model, tokenizer, sentences, label_ids, device):
    texts = [TEMPLATE.format(sentence=s, mask=tokenizer.mask_token) for s in sentences]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(**enc).logits
    mask_logits = logits[enc.input_ids == tokenizer.mask_token_id]
    return torch.stack([mask_logits[:, label_ids[0]], mask_logits[:, label_ids[1]]], dim=1)


@torch.no_grad()
def evaluate(model, tokenizer, data, label_ids, device, batch_size=64):
    model.eval()
    correct, total_loss, n = 0, 0.0, 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        scores = forward_logits(model, tokenizer, [x["sentence"] for x in batch], label_ids, device)
        labels = torch.tensor([x["label"] for x in batch], device=device)
        total_loss += F.cross_entropy(scores, labels, reduction="sum").item()
        correct += (scores.argmax(dim=1) == labels).sum().item()
        n += len(batch)
    return 100 * correct / n, total_loss / n


def train_one(seed, lr, batch_size, tokenizer, label_ids, device):
    torch.manual_seed(seed)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    train, val, test = load_cr_fewshot(k=16, seed=seed)
    optimizer = AdamW(model.parameters(), lr=lr)
    best = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(train)).tolist()
        for i in range(0, len(train), batch_size):
            batch = [train[j] for j in order[i:i + batch_size]]
            scores = forward_logits(model, tokenizer, [x["sentence"] for x in batch], label_ids, device)
            labels = torch.tensor([x["label"] for x in batch], device=device)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        val_acc, val_loss = evaluate(model, tokenizer, val, label_ids, device)
        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            test_acc = evaluate(model, tokenizer, test, label_ids, device)[0]
            best = {"val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc}
    del model; torch.cuda.empty_cache()
    return best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | CR prompt-based FT manuel | {len(SEEDS)} seeds")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label_ids = get_label_token_ids(tokenizer)

    per_seed = []
    test_scores = []
    for seed in SEEDS:
        best_cfg = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None, "cfg": None}
        for (lr, bs) in GRID:
            r = train_one(seed, lr, bs, tokenizer, label_ids, device)
            if r["val_acc"] > best_cfg["val_acc"] or (r["val_acc"] == best_cfg["val_acc"] and r["val_loss"] < best_cfg["val_loss"]):
                best_cfg = {**r, "cfg": (lr, bs)}
            print(f"  seed {seed} | lr={lr} bs={bs} -> val {r['val_acc']:.1f}% | test {r['test_acc']:.1f}%")
        print(f"==> seed {seed} : config {best_cfg['cfg']} -> TEST {best_cfg['test_acc']:.1f}%\n")
        per_seed.append({"seed": seed, "config": list(best_cfg["cfg"]), "test_acc": round(best_cfg["test_acc"], 1)})
        test_scores.append(best_cfg["test_acc"])

    mean, std = statistics.mean(test_scores), statistics.pstdev(test_scores)
    report = {
        "step": "04_prompt_ft_manual",
        "dataset": "CR (SetFit/SentEval-CR)",
        "method": "prompt-based fine-tuning (manuel)",
        "template": TEMPLATE, "label_words": LABEL_WORDS,
        "protocol": {"seeds": SEEDS, "grid": [list(g) for g in GRID], "epochs": EPOCHS},
        "per_seed": per_seed,
        "test_scores": [round(s, 1) for s in test_scores],
        "mean": round(mean, 1), "std": round(std, 1),
        "paper_target": PAPER_TARGET,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "04_prompt_ft_manual.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "04_prompt_ft_manual.txt"), "w") as f:
        f.write("ETAPE 4 — PROMPT-BASED FINE-TUNING (manuel) sur CR\n")
        f.write("=" * 50 + "\n")
        f.write(f"Template : {TEMPLATE} | mots-labels : {LABEL_WORDS}\n")
        f.write(f"Protocole : {len(SEEDS)} seeds x grille {GRID}, {EPOCHS} epochs\n\n")
        for p in per_seed:
            f.write(f"  seed {p['seed']:3d} (config {p['config']}) -> TEST {p['test_acc']:.1f}%\n")
        f.write(f"\nScores test : {[round(s,1) for s in test_scores]}\n")
        f.write(f"Prompt-based FT (manuel) : {mean:.1f} +/- {std:.1f}\n")
        f.write(f"Cible papier (Table 3, CR) : {PAPER_TARGET}\n")

    print("=" * 50)
    print(f"Scores test par seed : {[round(s, 1) for s in test_scores]}")
    print(f"Prompt-based FT (manuel) CR : {mean:.1f} +/- {std:.1f}")
    print(f"Cible papier : {PAPER_TARGET}")
    print(f"Rapport -> {OUT_DIR}/04_prompt_ft_manual.json et .txt")


if __name__ == "__main__":
    main()
