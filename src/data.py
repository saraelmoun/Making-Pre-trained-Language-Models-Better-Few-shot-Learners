"""
Chargement et echantillonnage few-shot de SST-2 (LM-BFF, Gao et al. 2021).

Trois ensembles, conformement au protocole du papier :
  - train few-shot : K=16 exemples PAR CLASSE, tires du gros train set
  - val (D_val)    : K=16 par classe AUSSI, tires du train set, DISJOINTS du train
                     (le papier l'appelle "val"/"valelopment" ; on dit "val" = validation,
                      meme chose : sert a regler les choix sans toucher au test)
  - test           : le split "validation" officiel de SST-2 (~872 phrases, labels publics)
                     -> c'est sur lui qu'on mesure les scores (le vrai test GLUE est cache)

Le tirage est controle par une graine (seed) : 5 graines = 5 splits (protocole du papier).
"""

import random
from datasets import load_dataset

# SST-2 dans GLUE : label 1 = positif, 0 = negatif
LABELS = {0: "negatif", 1: "positif"}


def load_sst2_fewshot(k=16, seed=42):
    """Renvoie (train, val, test) ou train/val = listes de k*2 exemples, test = validation complete.

    Chaque exemple est un dict {"sentence": str, "label": int}.
    """
    ds = load_dataset("nyu-mll/glue", "sst2")

    train_full = ds["train"]
    test = [{"sentence": x["sentence"].strip(), "label": x["label"]}
            for x in ds["validation"]]

    # Indices separes par classe
    idx_by_class = {0: [], 1: []}
    for i, lab in enumerate(train_full["label"]):
        idx_by_class[lab].append(i)

    rng = random.Random(seed)
    train, val = [], []
    for cls, idxs in idx_by_class.items():
        rng.shuffle(idxs)
        chosen = idxs[: 2 * k]          # 2k : la moitie pour train, l'autre pour val
        train_idx, val_idx = chosen[:k], chosen[k:2 * k]
        for i in train_idx:
            train.append({"sentence": train_full[i]["sentence"].strip(), "label": cls})
        for i in val_idx:
            val.append({"sentence": train_full[i]["sentence"].strip(), "label": cls})

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val, test


if __name__ == "__main__":
    # Petit controle visuel
    train, val, test = load_sst2_fewshot(k=16, seed=42)
    print(f"train : {len(train)}  | val : {len(val)}  | test : {len(test)}")
    n_pos = sum(x["label"] == 1 for x in train)
    print(f"train -> {n_pos} positifs / {len(train) - n_pos} negatifs")
    print("Exemple train :", train[0])
    print("Exemple test  :", test[0])
