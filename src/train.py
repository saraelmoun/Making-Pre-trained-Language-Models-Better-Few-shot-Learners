"""
Prompt-based FINE-TUNING (manuel) sur SST-2 — protocole complet du papier LM-BFF.

Par rapport a la version simple, on ajoute le protocole de robustesse du papier :
  - GRILLE d'hyperparametres (lr x batch) : val choisit la meilleure config
    (c'est la ou val est DISCRIMINANTE, cf. choix "grossiers")
  - selection du modele = meilleur val accuracy, DEPARTAGE par la perte val
    (l'accuracy val sature a 100% ; la perte val, elle, discrimine encore)
  - 5 SEEDS (5 splits) : on moyenne -> moyenne +/- ecart-type
    (le bruit du choix fin est absorbe par la moyenne sur les seeds)

On vise la ligne "Prompt-based FT (man)" de la Table 3 : ~92.7%.
Le template reste IMPOSE a la main (= "manuel").
"""

import argparse
import statistics
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data import load_sst2_fewshot

MODEL_NAME = "roberta-large"
TEMPLATE = "{sentence} It was {mask} ."
LABEL_WORDS = {1: "great", 0: "terrible"}

SEEDS = [13, 21, 42, 87, 100]                  # 5 splits, comme le papier
GRID = [(1e-5, 2), (1e-5, 4), (2e-5, 4), (2e-5, 8)]   # (lr, batch_size)
EPOCHS = 60


def get_label_token_ids(tokenizer):
    return {cls: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + w)[0])
            for cls, w in LABEL_WORDS.items()}


def forward_logits(model, tokenizer, sentences, label_ids, device):
    """Scores [B, 2] : colonne 0 = negatif, colonne 1 = positif (logique du zero-shot)."""
    texts = [TEMPLATE.format(sentence=s, mask=tokenizer.mask_token) for s in sentences]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(**enc).logits
    mask_pos = (enc.input_ids == tokenizer.mask_token_id)
    mask_logits = logits[mask_pos]
    return torch.stack([mask_logits[:, label_ids[0]], mask_logits[:, label_ids[1]]], dim=1)


@torch.no_grad()
def evaluate(model, tokenizer, data, label_ids, device, batch_size=64):
    """Renvoie (accuracy %, perte moyenne) sur 'data'."""
    model.eval()
    correct, total_loss, n = 0, 0.0, 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        scores = forward_logits(model, tokenizer,
                                [x["sentence"] for x in batch], label_ids, device)
        labels = torch.tensor([x["label"] for x in batch], device=device)
        total_loss += F.cross_entropy(scores, labels, reduction="sum").item()
        correct += (scores.argmax(dim=1) == labels).sum().item()
        n += len(batch)
    return 100 * correct / n, total_loss / n


def train_one(seed, lr, batch_size, tokenizer, label_ids, device):
    """Un entrainement complet pour (seed, lr, batch). Renvoie (val_acc, val_loss, test_acc)."""
    torch.manual_seed(seed)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    train, val, test = load_sst2_fewshot(k=16, seed=seed)
    optimizer = AdamW(model.parameters(), lr=lr)

    # selection : meilleur val_acc, departage par val_loss la plus faible
    best = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        for i in range(0, len(train), batch_size):
            batch = [train[j] for j in order[i:i + batch_size]]
            scores = forward_logits(model, tokenizer,
                                    [x["sentence"] for x in batch], label_ids, device)
            labels = torch.tensor([x["label"] for x in batch], device=device)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        val_acc, val_loss = evaluate(model, tokenizer, val, label_ids, device)
        better = (val_acc > best["val_acc"] or
                  (val_acc == best["val_acc"] and val_loss < best["val_loss"]))
        if better:
            test_acc, _ = evaluate(model, tokenizer, test, label_ids, device)
            best = {"val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc}

    del model
    torch.cuda.empty_cache()
    return best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | {len(SEEDS)} seeds | grille : {GRID} | epochs : {EPOCHS}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label_ids = get_label_token_ids(tokenizer)

    test_scores = []
    for seed in SEEDS:
        # pour ce seed : on essaie toute la grille, val choisit la meilleure config
        best_cfg = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None, "cfg": None}
        for (lr, bs) in GRID:
            r = train_one(seed, lr, bs, tokenizer, label_ids, device)
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
    print(f"Prompt-based FT (manuel) : {mean:.1f} +/- {std:.1f}")
    print(f"Chiffre du papier (Table 3) : 92.7 +/- 0.9")
    print("=" * 50)


if __name__ == "__main__":
    main()
