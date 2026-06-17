"""
Chargement few-shot de CR (Customer Reviews) — fichier NEUF et isole.
SetFit/SentEval-CR : sentiment binaire, 0 = negatif, 1 = positif.
Pas de split validation -> val tiree depuis le train ; test = split test (753).

Lance en script : produit 02_data_loading.json / .txt
"""

import json
import os
import random
from datasets import load_dataset

HF_NAME = "SetFit/SentEval-CR"
LABELS = {0: "negatif", 1: "positif"}
OUT_DIR = "outputs_cr"


def load_cr_fewshot(k=16, seed=42):
    """Renvoie (train, val, test). train/val = k*2 exemples (tires du train) ; test = split test."""
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
        chosen = idxs[: 2 * k]
        tr, va = chosen[:k], chosen[k:2 * k]
        for i in tr:
            train.append({"sentence": train_full[i]["text"].strip(), "label": cls})
        for i in va:
            val.append({"sentence": train_full[i]["text"].strip(), "label": cls})
    rng.shuffle(train); rng.shuffle(val)
    return train, val, test


def main():
    train, val, test = load_cr_fewshot(k=16, seed=42)
    n_pos = sum(x["label"] == 1 for x in train)
    report = {
        "step": "02_data_loading", "dataset": "CR (SetFit/SentEval-CR)",
        "k_per_class": 16, "seed": 42,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "train_balance": {"positif": n_pos, "negatif": len(train) - n_pos},
        "example_train": train[0], "example_test": test[0],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "02_data_loading.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "02_data_loading.txt"), "w") as f:
        f.write("ETAPE 2 — CHARGEMENT FEW-SHOT DE CR\n" + "=" * 40 + "\n")
        f.write(f"Dataset : {report['dataset']}\nK par classe : 16 (seed 42)\n\n")
        f.write(f"Tailles : train {len(train)} | val {len(val)} | test {len(test)}\n")
        f.write(f"Equilibre train : {n_pos} positifs / {len(train)-n_pos} negatifs\n\n")
        f.write(f"Exemple train : {train[0]}\nExemple test  : {test[0]}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRapport -> {OUT_DIR}/02_data_loading.json et .txt")


if __name__ == "__main__":
    main()
