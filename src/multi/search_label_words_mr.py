"""
ETAPE 7 — Auto-selection des MOTS-LABELS (§5.1) sur MR (fichier NEUF, independant).

Le template est FIXE ("{sentence} It was [MASK] ."), on cherche les MOTS qui
representent le mieux chaque classe. Tache binaire (MR : 0=negatif, 1=positif).

Chaque sous-etape est sauvegardee :
  07a_label_candidates.json/.txt    (top-k mots par classe, eq. 3)
  07b_zeroshot_ranking.json         (combinaisons classees par accuracy zero-shot)
  07c_label_selection.json          (finalistes fine-tunes, classes sur val)
  07_auto_labelwords.json/.txt      (gagnant + eval 5 seeds)

N'importe QUE data_mr. Ne touche pas au code SST-2.
"""

import json
import os
import statistics
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data_mr import load_mr_fewshot

transformers.logging.set_verbosity_error()

MODEL_NAME = "roberta-large"
TEMPLATE = "{sentence} It was {mask} ."     # template FIXE (manuel)
SEEDS = [13, 21, 42, 87, 100]
GRID = [(1e-5, 2), (1e-5, 4), (2e-5, 4), (2e-5, 8)]
EPOCHS = 60
K = 100            # mots candidats par classe
N_FINALISTS = 10   # combinaisons gardees apres filtre zero-shot
RANK_SEED = 42
OUT_DIR = "outputs_mr"


# ---------------------------------------------------------------------------
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


def reset_weights(model, init_state):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.copy_(init_state[k])


def train_eval(model, tokenizer, label_ids, seed, lr, batch_size, epochs, device, eval_test, init_state):
    reset_weights(model, init_state)
    torch.manual_seed(seed)
    train, val, test = load_mr_fewshot(k=16, seed=seed)
    optimizer = AdamW(model.parameters(), lr=lr)
    best = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
    for _ in range(epochs):
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
            test_acc = evaluate(model, tokenizer, test, label_ids, device)[0] if eval_test else None
            best = {"val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc}
    return best


@torch.no_grad()
def class_topk_words(model, tokenizer, train, k, device, batch_size=16):
    """Top-k mots 'propres' par classe (eq. 3)."""
    mask_id = tokenizer.mask_token_id
    per_class = {}
    for cls in sorted({x["label"] for x in train}):
        exs = [x["sentence"] for x in train if x["label"] == cls]
        sum_logp = None
        for i in range(0, len(exs), batch_size):
            texts = [TEMPLATE.format(sentence=s, mask=tokenizer.mask_token) for s in exs[i:i + batch_size]]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            logits = model(**enc).logits
            logp = torch.log_softmax(logits[enc.input_ids == mask_id], dim=-1).sum(0)
            sum_logp = logp if sum_logp is None else sum_logp + logp
        mean_logp = sum_logp / len(exs)
        topv, topi = mean_logp.topk(k * 30)
        cands = []
        for score, tid in zip(topv.tolist(), topi.tolist()):
            tok = tokenizer.convert_ids_to_tokens(tid)
            word = tok.lstrip("Ġ")
            if tok.startswith("Ġ") and word.isalpha() and len(word) > 1:
                cands.append((tid, word.lower(), round(score, 4)))
            if len(cands) >= k:
                break
        per_class[cls] = cands
    return per_class


@torch.no_grad()
def zeroshot_accuracy_matrix(model, tokenizer, train, ids0, ids1, device, batch_size=16):
    mask_id = tokenizer.mask_token_id
    all_logits, gold = [], []
    for i in range(0, len(train), batch_size):
        batch = train[i:i + batch_size]
        texts = [TEMPLATE.format(sentence=x["sentence"], mask=tokenizer.mask_token) for x in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        all_logits.append(model(**enc).logits[enc.input_ids == mask_id])
        gold += [x["label"] for x in batch]
    M = torch.cat(all_logits, dim=0)
    gold = torch.tensor(gold, device=device)
    A, B = M[:, ids0], M[:, ids1]
    pred1 = A.unsqueeze(2) < B.unsqueeze(1)
    gold1 = (gold == 1).view(-1, 1, 1)
    return (pred1 == gold1).float().mean(0)        # [K0, K1]


def save(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | MR §5.1 auto mots-labels | template fixe : {TEMPLATE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train, _, _ = load_mr_fewshot(k=16, seed=RANK_SEED)

    # ---- 07a : top-k par classe ----
    print(f"\n[1] Top-{K} mots par classe (seed {RANK_SEED})...")
    per_class = class_topk_words(model, tokenizer, train, K, device)
    save("07a_label_candidates.json",
         {str(c): [{"token_id": t, "word": w, "score": s} for t, w, s in lst]
          for c, lst in per_class.items()})
    with open(os.path.join(OUT_DIR, "07a_label_candidates.txt"), "w") as f:
        for c, lst in per_class.items():
            f.write(f"classe {c} : " + ", ".join(w for _, w, _ in lst[:30]) + " ...\n")
    for c, lst in per_class.items():
        print(f"  classe {c}: " + ", ".join(w for _, w, _ in lst[:12]) + " ...")

    # ---- 07b : filtre zero-shot ----
    ids0 = [t for t, _, _ in per_class[0]]; words0 = [w for _, w, _ in per_class[0]]
    ids1 = [t for t, _, _ in per_class[1]]; words1 = [w for _, w, _ in per_class[1]]
    print(f"\n[2] Filtre zero-shot sur {len(ids0)}x{len(ids1)} combinaisons...")
    acc = zeroshot_accuracy_matrix(model, tokenizer, train, ids0, ids1, device).view(-1)
    order = acc.argsort(descending=True)
    ranking = []
    for idx in order.tolist():
        i, j = divmod(idx, len(ids1))
        ranking.append({"class0_word": words0[i], "class1_word": words1[j],
                        "class0_id": ids0[i], "class1_id": ids1[j],
                        "zeroshot_acc": round(acc[idx].item() * 100, 2)})
    save("07b_zeroshot_ranking.json", ranking[:100])
    finalists = ranking[:N_FINALISTS]
    print("  Top 5 (zero-shot) :")
    for r in finalists[:5]:
        print(f"    {{0:{r['class0_word']}, 1:{r['class1_word']}}} -> {r['zeroshot_acc']:.1f}%")

    # ---- 07c : fine-tuning des finalistes ----
    print(f"\n[3] Fine-tuning des {len(finalists)} finalistes...")
    selection = []
    for r in finalists:
        res = train_eval(model, tokenizer, {0: r["class0_id"], 1: r["class1_id"]},
                         RANK_SEED, 1e-5, 4, EPOCHS, device, eval_test=False, init_state=init_state)
        selection.append({**r, "val_acc": res["val_acc"], "val_loss": round(res["val_loss"], 4)})
        print(f"    {{0:{r['class0_word']}, 1:{r['class1_word']}}} -> val {res['val_acc']:.1f}%")
    selection.sort(key=lambda x: (x["val_acc"], -x["val_loss"]), reverse=True)
    save("07c_label_selection.json", selection)
    best = selection[0]
    print(f"\n>>> MEILLEURS mots-labels : {{0:{best['class0_word']}, 1:{best['class1_word']}}}")

    # ---- 07 : eval finale 5 seeds ----
    print("\n[4] Evaluation finale (5 seeds x grille)...")
    best_ids = {0: best["class0_id"], 1: best["class1_id"]}
    test_scores = []
    for seed in SEEDS:
        bc = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
        for (lr, bs) in GRID:
            res = train_eval(model, tokenizer, best_ids, seed, lr, bs, EPOCHS, device, True, init_state)
            if res["val_acc"] > bc["val_acc"] or (res["val_acc"] == bc["val_acc"] and res["val_loss"] < bc["val_loss"]):
                bc = res
        print(f"    seed {seed} -> TEST {bc['test_acc']:.1f}%")
        test_scores.append(bc["test_acc"])
    mean, std = statistics.mean(test_scores), statistics.pstdev(test_scores)

    result = {"step": "07_auto_labelwords", "dataset": "MR (rotten_tomatoes)",
              "best_label_words": {"0": best["class0_word"], "1": best["class1_word"]},
              "test_scores": [round(s, 1) for s in test_scores],
              "mean": round(mean, 1), "std": round(std, 1),
              "note_manuel_nous": "great/terrible -> 85.4 +/- 2.7"}
    save("07_auto_labelwords.json", result)
    with open(os.path.join(OUT_DIR, "07_auto_labelwords.txt"), "w") as f:
        f.write("ETAPE 7 — AUTO MOTS-LABELS (§5.1) sur MR\n" + "=" * 45 + "\n")
        f.write(f"Mots-labels auto : {{0:{best['class0_word']}, 1:{best['class1_word']}}}\n")
        f.write(f"Scores test : {[round(s,1) for s in test_scores]}\n")
        f.write(f"Prompt-based FT (AUTO label words) : {mean:.1f} +/- {std:.1f}\n")
        f.write(f"(rappel) manuel great/terrible, nous : 85.4 +/- 2.7\n")

    print("\n" + "=" * 50)
    print(f"Mots-labels auto MR : {{0:{best['class0_word']}, 1:{best['class1_word']}}}")
    print(f"Prompt-based FT (AUTO label words) : {mean:.1f} +/- {std:.1f}")
    print(f"(rappel) manuel, nous : 85.4 +/- 2.7")
    print("=" * 50)


if __name__ == "__main__":
    main()
