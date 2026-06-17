"""
Baseline : FINE-TUNING CLASSIQUE (standard) sur SST-2 — LM-BFF, Gao et al. 2021.

C'est la methode "d'avant les prompts", a comparer au prompt-based.
Difference cle avec train.py :
  - PAS de template, PAS de [MASK], PAS de mots-labels.
  - On ajoute une TETE DE CLASSIFICATION neuve (matrice W initialisee au HASARD)
    par-dessus le vecteur [CLS]. Pour SST-2 binaire : 2 x 1024 = 2048 parametres neufs.
  - On apprend ces 2048 parametres a partir de ZERO avec seulement 32 exemples.

But : montrer que cette approche ECHOUE en few-shot (faible accuracy, forte instabilite),
la ou le prompt-based reussit. On vise la ligne "Fine-tuning" de la Table 3 : ~81.4 +/- 3.8.

Meme protocole que train.py : 5 seeds, grille (lr x batch), selection sur val, moyenne +/- ecart-type.
"""

import statistics
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data import load_sst2_fewshot

transformers.logging.set_verbosity_error()   # silence le LOAD REPORT verbeux

MODEL_NAME = "roberta-large"
SEEDS = [13, 21, 42, 87, 100]
GRID = [(1e-5, 2), (1e-5, 4), (2e-5, 4), (2e-5, 8)]
EPOCHS = 60


@torch.no_grad()
def evaluate(model, tokenizer, data, device, batch_size=64):
    """Renvoie (accuracy %, perte moyenne). Ici le modele sort directement 2 logits."""
    model.eval()
    correct, total_loss, n = 0, 0.0, 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        enc = tokenizer([x["sentence"] for x in batch], return_tensors="pt",
                        padding=True, truncation=True).to(device)
        logits = model(**enc).logits                  # [B, 2]  <- tete de classif
        labels = torch.tensor([x["label"] for x in batch], device=device)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += len(batch)
    return 100 * correct / n, total_loss / n


def train_one(seed, lr, batch_size, tokenizer, device):
    torch.manual_seed(seed)
    # La tete de classification (2 classes) est ajoutee et initialisee AU HASARD ici :
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(device)
    train, val, test = load_sst2_fewshot(k=16, seed=seed)
    optimizer = AdamW(model.parameters(), lr=lr)

    best = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        for i in range(0, len(train), batch_size):
            batch = [train[j] for j in order[i:i + batch_size]]
            enc = tokenizer([x["sentence"] for x in batch], return_tensors="pt",
                            padding=True, truncation=True).to(device)
            labels = torch.tensor([x["label"] for x in batch], device=device)
            loss = model(**enc, labels=labels).loss   # la tete calcule la perte
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        val_acc, val_loss = evaluate(model, tokenizer, val, device)
        better = (val_acc > best["val_acc"] or
                  (val_acc == best["val_acc"] and val_loss < best["val_loss"]))
        if better:
            test_acc, _ = evaluate(model, tokenizer, test, device)
            best = {"val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc}

    del model
    torch.cuda.empty_cache()
    return best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | BASELINE fine-tuning classique | {len(SEEDS)} seeds")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_scores = []
    for seed in SEEDS:
        best_cfg = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None, "cfg": None}
        for (lr, bs) in GRID:
            r = train_one(seed, lr, bs, tokenizer, device)
            chosen = (r["val_acc"] > best_cfg["val_acc"] or
                      (r["val_acc"] == best_cfg["val_acc"] and r["val_loss"] < best_cfg["val_loss"]))
            if chosen:
                best_cfg = {**r, "cfg": (lr, bs)}
            print(f"  seed {seed} | lr={lr} bs={bs} -> val {r['val_acc']:.1f}% "
                  f"(loss {r['val_loss']:.3f}) | test {r['test_acc']:.1f}%")
        print(f"==> seed {seed} : config retenue {best_cfg['cfg']} -> TEST {best_cfg['test_acc']:.1f}%\n")
        test_scores.append(best_cfg["test_acc"])

    mean = statistics.mean(test_scores)
    std = statistics.pstdev(test_scores)
    print("=" * 50)
    print(f"Scores test par seed : {[round(s, 1) for s in test_scores]}")
    print(f"Fine-tuning CLASSIQUE : {mean:.1f} +/- {std:.1f}")
    print(f"Chiffre du papier (Table 3) : 81.4 +/- 3.8")
    print(f"(rappel) Prompt-based FT, nous : 89.2 +/- 1.0")
    print("=" * 50)


if __name__ == "__main__":
    main()
