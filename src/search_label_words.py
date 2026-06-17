"""
Auto-selection des MOTS-LABELS (LM-BFF, Gao et al. 2021, section 5.1).

Le template est FIXE ("{sentence} It was [MASK] ."), on cherche les MOTS qui
representent le mieux chaque classe. Pipeline en 5 etapes, chacune SAUVEGARDEE :

  1. TOP-K par classe (eq. 3) : pour chaque classe, on moyenne sur ses exemples la
     distribution du vocabulaire sous [MASK], et on garde les k meilleurs mots.
       -> label_candidates.json / .txt
  2. COMBINAISONS : produit des candidats des deux classes (k x k assignations).
  3. FILTRE ZERO-SHOT (rapide) : accuracy zero-shot de chaque combinaison sur le train,
     on garde les n meilleures. (vectorise, pas de fine-tuning)
       -> label_zeroshot_ranking.json
  4. FILTRE FINE-TUNING (precis) : on fine-tune les n finalistes, on classe sur val.
       -> label_selection_results.json
  5. EVALUATION FINALE du gagnant sur 5 seeds (comme le manuel).
       -> auto_labelword_result.json

NIVEAU 1 : recherche sur seed 42, puis evaluation du gagnant sur 5 seeds.
NB : ce script suppose une tache BINAIRE (SST-2 : 2 classes).
"""

import argparse
import json
import statistics
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data import load_sst2_fewshot

transformers.logging.set_verbosity_error()

MODEL_NAME = "roberta-large"
TEMPLATE = "{sentence} It was {mask} ."     # template FIXE (le manuel)
SEEDS = [13, 21, 42, 87, 100]
GRID = [(1e-5, 2), (1e-5, 4), (2e-5, 4), (2e-5, 8)]


# ----------------------------------------------------------------------------
# Briques reutilisees (identiques a select_templates.py)
# ----------------------------------------------------------------------------
def forward_logits(model, tokenizer, sentences, label_ids, device):
    """Scores [B, 2] (col 0 = classe 0, col 1 = classe 1) pour des mots-labels donnes."""
    texts = [TEMPLATE.format(sentence=s, mask=tokenizer.mask_token) for s in sentences]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(**enc).logits
    mask_pos = (enc.input_ids == tokenizer.mask_token_id)
    mask_logits = logits[mask_pos]
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
    train, val, test = load_sst2_fewshot(k=16, seed=seed)
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


# ----------------------------------------------------------------------------
# Etape 1 : top-k des mots par classe (eq. 3)
# ----------------------------------------------------------------------------
@torch.no_grad()
def class_topk_words(model, tokenizer, train, k, device, batch_size=16):
    """Pour chaque classe : moyenne sur ses exemples de log P([MASK]=v), top-k mots 'propres'.
    Renvoie {classe: [(token_id, mot, score), ...]} et la matrice des log-probs [n_train, vocab]."""
    mask_id = tokenizer.mask_token_id
    classes = sorted({x["label"] for x in train})
    per_class = {}
    for cls in classes:
        exs = [x["sentence"] for x in train if x["label"] == cls]
        sum_logp = None
        for i in range(0, len(exs), batch_size):
            batch = exs[i:i + batch_size]
            texts = [TEMPLATE.format(sentence=s, mask=tokenizer.mask_token) for s in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            logits = model(**enc).logits
            mask_logits = logits[enc.input_ids == mask_id]              # [B, vocab]
            logp = torch.log_softmax(mask_logits, dim=-1).sum(0)        # somme sur le batch
            sum_logp = logp if sum_logp is None else sum_logp + logp
        mean_logp = sum_logp / len(exs)                                 # moyenne sur la classe

        # on prend large puis on filtre aux mots "propres" (precedes d'un espace, alphabetiques)
        topv, topi = mean_logp.topk(k * 30)
        cands = []
        for score, tid in zip(topv.tolist(), topi.tolist()):
            tok = tokenizer.convert_ids_to_tokens(tid)
            word = tok.lstrip("Ġ")                                 # 'Ġ' = espace en RoBERTa
            if tok.startswith("Ġ") and word.isalpha() and len(word) > 1:
                cands.append((tid, word.lower(), round(score, 4)))
            if len(cands) >= k:
                break
        per_class[cls] = cands
    return per_class


# ----------------------------------------------------------------------------
# Etape 3 : filtre zero-shot vectorise (tache binaire)
# ----------------------------------------------------------------------------
@torch.no_grad()
def zeroshot_accuracy_matrix(model, tokenizer, train, ids0, ids1, device, batch_size=16):
    """Renvoie une matrice acc[i, j] = accuracy zero-shot de l'assignation
    (classe0 -> ids0[i], classe1 -> ids1[j]) sur le train. Vectorise."""
    mask_id = tokenizer.mask_token_id
    all_mask_logits, gold = [], []
    for i in range(0, len(train), batch_size):
        batch = train[i:i + batch_size]
        texts = [TEMPLATE.format(sentence=x["sentence"], mask=tokenizer.mask_token) for x in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        logits = model(**enc).logits
        all_mask_logits.append(logits[enc.input_ids == mask_id])       # [B, vocab]
        gold += [x["label"] for x in batch]
    M = torch.cat(all_mask_logits, dim=0)                              # [n_train, vocab]
    gold = torch.tensor(gold, device=device)

    A = M[:, ids0]                                                     # [n, K0] (classe 0)
    B = M[:, ids1]                                                     # [n, K1] (classe 1)
    # prediction = classe 1 si score(mot1) > score(mot0)
    pred1 = A.unsqueeze(2) < B.unsqueeze(1)                            # [n, K0, K1] bool
    gold1 = (gold == 1).view(-1, 1, 1)
    acc = (pred1 == gold1).float().mean(0)                            # [K0, K1]
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100, help="mots candidats par classe (top-k)")
    ap.add_argument("--n_finalists", type=int, default=10, help="combinaisons gardees apres zero-shot")
    ap.add_argument("--rank_seed", type=int, default=42)
    ap.add_argument("--rank_lr", type=float, default=1e-5)
    ap.add_argument("--rank_bs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | template fixe : {TEMPLATE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    train, _, _ = load_sst2_fewshot(k=16, seed=args.rank_seed)

    # ----- ETAPE 1 : top-k par classe -----
    print(f"\n[1] Top-{args.k} mots par classe (seed {args.rank_seed})...")
    per_class = class_topk_words(model, tokenizer, train, args.k, device)
    with open("label_candidates.json", "w") as f:
        json.dump({str(c): [{"token_id": t, "word": w, "score": s} for t, w, s in lst]
                   for c, lst in per_class.items()}, f, indent=2, ensure_ascii=False)
    with open("label_candidates.txt", "w") as f:
        for c, lst in per_class.items():
            f.write(f"classe {c} : " + ", ".join(w for _, w, _ in lst[:30]) + " ...\n")
    for c, lst in per_class.items():
        print(f"  classe {c}: " + ", ".join(w for _, w, _ in lst[:12]) + " ...")

    # ----- ETAPE 2+3 : combinaisons + filtre zero-shot -----
    ids0 = [t for t, _, _ in per_class[0]]
    ids1 = [t for t, _, _ in per_class[1]]
    words0 = [w for _, w, _ in per_class[0]]
    words1 = [w for _, w, _ in per_class[1]]
    print(f"\n[3] Filtre zero-shot sur {len(ids0)}x{len(ids1)} = {len(ids0)*len(ids1)} combinaisons...")
    acc = zeroshot_accuracy_matrix(model, tokenizer, train, ids0, ids1, device)  # [K0,K1]
    flat = acc.view(-1)
    order = flat.argsort(descending=True)
    ranking = []
    for idx in order.tolist():
        i, j = divmod(idx, len(ids1))
        ranking.append({"class0_word": words0[i], "class1_word": words1[j],
                        "class0_id": ids0[i], "class1_id": ids1[j],
                        "zeroshot_acc": round(flat[idx].item() * 100, 2)})
    with open("label_zeroshot_ranking.json", "w") as f:
        json.dump(ranking[:100], f, indent=2, ensure_ascii=False)     # on garde le top 100
    finalists = ranking[:args.n_finalists]
    print("  Top 5 combinaisons (zero-shot) :")
    for r in finalists[:5]:
        print(f"    {{0:{r['class0_word']}, 1:{r['class1_word']}}} -> {r['zeroshot_acc']:.1f}%")

    # ----- ETAPE 4 : fine-tuning des finalistes, classement sur val -----
    print(f"\n[4] Fine-tuning des {len(finalists)} finalistes (seed {args.rank_seed})...")
    selection = []
    for r in finalists:
        label_ids = {0: r["class0_id"], 1: r["class1_id"]}
        res = train_eval(model, tokenizer, label_ids, args.rank_seed, args.rank_lr, args.rank_bs,
                         args.epochs, device, eval_test=False, init_state=init_state)
        selection.append({**r, "val_acc": res["val_acc"], "val_loss": round(res["val_loss"], 4)})
        print(f"    {{0:{r['class0_word']}, 1:{r['class1_word']}}} -> val {res['val_acc']:.1f}% "
              f"(loss {res['val_loss']:.3f})")
    selection.sort(key=lambda x: (x["val_acc"], -x["val_loss"]), reverse=True)
    with open("label_selection_results.json", "w") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)
    best = selection[0]
    print(f"\n>>> MEILLEURS mots-labels : {{0:{best['class0_word']}, 1:{best['class1_word']}}}")

    # ----- ETAPE 5 : evaluation finale du gagnant (5 seeds x grille) -----
    print("\n[5] Evaluation finale (5 seeds x grille)...")
    best_ids = {0: best["class0_id"], 1: best["class1_id"]}
    test_scores = []
    for seed in SEEDS:
        bc = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
        for (lr, bs) in GRID:
            res = train_eval(model, tokenizer, best_ids, seed, lr, bs, args.epochs,
                             device, eval_test=True, init_state=init_state)
            if res["val_acc"] > bc["val_acc"] or (res["val_acc"] == bc["val_acc"] and res["val_loss"] < bc["val_loss"]):
                bc = res
        print(f"    seed {seed} -> TEST {bc['test_acc']:.1f}%")
        test_scores.append(bc["test_acc"])
    mean, std = statistics.mean(test_scores), statistics.pstdev(test_scores)

    result = {"best_label_words": {"0": best["class0_word"], "1": best["class1_word"]},
              "test_scores": test_scores, "mean": mean, "std": std}
    with open("auto_labelword_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"Mots-labels auto : {{0:{best['class0_word']}, 1:{best['class1_word']}}}")
    print(f"Scores test par seed : {[round(s, 1) for s in test_scores]}")
    print(f"Prompt-based FT (AUTO label words) : {mean:.1f} +/- {std:.1f}")
    print(f"(rappel) manuel great/terrible, nous : 89.2 +/- 1.0 | papier auto-L (Table 5) : 91.5")
    print("=" * 50)


if __name__ == "__main__":
    main()
