"""
Chargement few-shot de MR (Movie Reviews) — fichier NEUF et isole.
Ne touche PAS au data.py de SST-2. Calque sur data.py mais pour MR.

MR (rotten_tomatoes) : sentiment binaire, 0 = negatif, 1 = positif.
On tire K=16 exemples PAR CLASSE pour train ET pour val (disjoints),
le test = le split test officiel (1066 phrases).

Lance en script : produit le rapport d'etape 02_data_loading.json / .txt
"""

import json
import os
import random
from datasets import load_dataset

HF_NAME = "cornell-movie-review-data/rotten_tomatoes"
LABELS = {0: "negatif", 1: "positif"}
OUT_DIR = "outputs_mr"   # rempli localement apres rapatriement


def load_mr_fewshot(k=16, seed=42):
    """Renvoie (train, val, test). train/val = k*2 exemples ; test = split test complet.
    Chaque exemple = {"sentence": str, "label": int}."""
    ds = load_dataset(HF_NAME)
    train_full = ds["train"]
    test = [{"sentence": x["text"].strip(), "label": x["label"]} for x in ds["test"]]

    idx_by_class = {0: [], 1: []}
    for i, lab in enumerate(train_full["label"]):
        idx_by_class[lab].append(i)

    rng = random.Random(seed)
    train, val = [], []
    for cls, idxs in idx_by_class.items():
        rng.shuffle(idxs)
        chosen = idxs[: 2 * k]                  # moitie train, moitie val
        tr, va = chosen[:k], chosen[k:2 * k]
        for i in tr:
            train.append({"sentence": train_full[i]["text"].strip(), "label": cls})
        for i in va:
            val.append({"sentence": train_full[i]["text"].strip(), "label": cls})
    rng.shuffle(train); rng.shuffle(val)
    return train, val, test


def main():
    train, val, test = load_mr_fewshot(k=16, seed=42)
    n_pos = sum(x["label"] == 1 for x in train)

    report = {
        "step": "02_data_loading",
        "dataset": "MR (rotten_tomatoes)",
        "k_per_class": 16,
        "seed": 42,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "train_balance": {"positif": n_pos, "negatif": len(train) - n_pos},
        "example_train": train[0],
        "example_test": test[0],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "02_data_loading.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "02_data_loading.txt"), "w") as f:
        f.write("ETAPE 2 — CHARGEMENT FEW-SHOT DE MR\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset      : {report['dataset']}\n")
        f.write(f"K par classe : 16 (seed 42)\n\n")
        f.write(f"Tailles : train {len(train)} | val {len(val)} | test {len(test)}\n")
        f.write(f"Equilibre train : {n_pos} positifs / {len(train)-n_pos} negatifs\n\n")
        f.write(f"Exemple train : {train[0]}\n")
        f.write(f"Exemple test  : {test[0]}\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport sauvegarde dans {OUT_DIR}/02_data_loading.json et .txt")


if __name__ == "__main__":
    main()
