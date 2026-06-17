"""
ETAPE 5 — Baseline FINE-TUNING CLASSIQUE sur MR (fichier NEUF, isole).

Methode "d'avant les prompts" : tete de classification neuve (initialisee au hasard)
au-dessus de [CLS], apprise a partir de zero sur 32 exemples. A comparer au prompt-based.
Meme protocole : 5 seeds x grille, selection sur val. Cible papier (Table 3, MR) : 76.9.
Sauvegarde 05_baseline.json / .txt.
"""

import json
import os
import statistics
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_mr import load_mr_fewshot

transformers.logging.set_verbosity_error()

MODEL_NAME = "roberta-large"
SEEDS = [13, 21, 42, 87, 100]
GRID = [(1e-5, 2), (1e-5, 4), (2e-5, 4), (2e-5, 8)]
EPOCHS = 60
PAPER_TARGET = 76.9
OUT_DIR = "outputs_mr"


@torch.no_grad()
def evaluate(model, tokenizer, data, device, batch_size=64):
    model.eval()
    correct, total_loss, n = 0, 0.0, 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        enc = tokenizer([x["sentence"] for x in batch], return_tensors="pt",
                        padding=True, truncation=True).to(device)
        logits = model(**enc).logits
        labels = torch.tensor([x["label"] for x in batch], device=device)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += len(batch)
    return 100 * correct / n, total_loss / n


def train_one(seed, lr, batch_size, tokenizer, device):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    train, val, test = load_mr_fewshot(k=16, seed=seed)
    optimizer = AdamW(model.parameters(), lr=lr)
    best = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(train)).tolist()
        for i in range(0, len(train), batch_size):
            batch = [train[j] for j in order[i:i + batch_size]]
            enc = tokenizer([x["sentence"] for x in batch], return_tensors="pt",
                            padding=True, truncation=True).to(device)
            labels = torch.tensor([x["label"] for x in batch], device=device)
            loss = model(**enc, labels=labels).loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        val_acc, val_loss = evaluate(model, tokenizer, val, device)
        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            test_acc = evaluate(model, tokenizer, test, device)[0]
            best = {"val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc}
    del model; torch.cuda.empty_cache()
    return best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | MR baseline classique | {len(SEEDS)} seeds")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    per_seed, test_scores = [], []
    for seed in SEEDS:
        best_cfg = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None, "cfg": None}
        for (lr, bs) in GRID:
            r = train_one(seed, lr, bs, tokenizer, device)
            if r["val_acc"] > best_cfg["val_acc"] or (r["val_acc"] == best_cfg["val_acc"] and r["val_loss"] < best_cfg["val_loss"]):
                best_cfg = {**r, "cfg": (lr, bs)}
            print(f"  seed {seed} | lr={lr} bs={bs} -> val {r['val_acc']:.1f}% | test {r['test_acc']:.1f}%")
        print(f"==> seed {seed} : config {best_cfg['cfg']} -> TEST {best_cfg['test_acc']:.1f}%\n")
        per_seed.append({"seed": seed, "config": list(best_cfg["cfg"]), "test_acc": round(best_cfg["test_acc"], 1)})
        test_scores.append(best_cfg["test_acc"])

    mean, std = statistics.mean(test_scores), statistics.pstdev(test_scores)
    report = {
        "step": "05_baseline",
        "dataset": "MR (rotten_tomatoes)",
        "method": "fine-tuning classique (tete de classification standard)",
        "protocol": {"seeds": SEEDS, "grid": [list(g) for g in GRID], "epochs": EPOCHS},
        "per_seed": per_seed,
        "test_scores": [round(s, 1) for s in test_scores],
        "mean": round(mean, 1), "std": round(std, 1),
        "paper_target": PAPER_TARGET,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "05_baseline.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "05_baseline.txt"), "w") as f:
        f.write("ETAPE 5 — BASELINE FINE-TUNING CLASSIQUE sur MR\n")
        f.write("=" * 50 + "\n")
        f.write(f"Protocole : {len(SEEDS)} seeds x grille {GRID}, {EPOCHS} epochs\n\n")
        for p in per_seed:
            f.write(f"  seed {p['seed']:3d} (config {p['config']}) -> TEST {p['test_acc']:.1f}%\n")
        f.write(f"\nScores test : {[round(s,1) for s in test_scores]}\n")
        f.write(f"Fine-tuning CLASSIQUE : {mean:.1f} +/- {std:.1f}\n")
        f.write(f"Cible papier (Table 3, MR) : {PAPER_TARGET}\n")

    print("=" * 50)
    print(f"Scores test par seed : {[round(s, 1) for s in test_scores]}")
    print(f"Fine-tuning CLASSIQUE MR : {mean:.1f} +/- {std:.1f}")
    print(f"Cible papier : {PAPER_TARGET}")
    print(f"Rapport -> {OUT_DIR}/05_baseline.json et .txt")


if __name__ == "__main__":
    main()
