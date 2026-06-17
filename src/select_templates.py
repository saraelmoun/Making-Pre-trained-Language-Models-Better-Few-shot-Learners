"""
SELECTION des templates auto-generes (LM-BFF, Gao et al. 2021, section 5.2).

Le score de generation T5 dit juste "ce template est linguistiquement probable",
PAS "il classe bien le sentiment". On tranche donc par l'experience :

  ETAPE A (classement) : pour CHAQUE template candidat (templates.json), on fine-tune
    RoBERTa sur les 32 exemples (seed 42) et on mesure sur VAL. On classe par accuracy
    val (departage par perte val). -> le meilleur template selon val.
    (val seulement = methode du papier ; on ne touche jamais au test pour choisir.)

  ETAPE B (evaluation finale) : le template GAGNANT est evalue avec le protocole complet
    (5 seeds x grille hyperparametres), exactement comme le fine-tuning manuel.
    -> chiffre final comparable au manuel (89.2 +/- 1.0).

NIVEAU 1 : on selectionne sur seed 42 (les templates ont ete generes depuis ce seed),
puis on evalue le gagnant sur 5 seeds. (Le niveau 2 = regenerer+selectionner par seed.)

Sortie : selection_results.json / .txt (classement complet des 159) + le resultat final.
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
LABEL_WORDS = {1: "great", 0: "terrible"}
SEEDS = [13, 21, 42, 87, 100]
GRID = [(1e-5, 2), (1e-5, 4), (2e-5, 4), (2e-5, 8)]


def get_label_token_ids(tokenizer):
    return {cls: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + w)[0])
            for cls, w in LABEL_WORDS.items()}


def forward_logits(model, tokenizer, sentences, template, label_ids, device):
    """Scores [B, 2] (col 0 = negatif, col 1 = positif) pour un template donne."""
    texts = [template.format(sentence=s, mask=tokenizer.mask_token) for s in sentences]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(**enc).logits
    mask_pos = (enc.input_ids == tokenizer.mask_token_id)
    mask_logits = logits[mask_pos]
    return torch.stack([mask_logits[:, label_ids[0]], mask_logits[:, label_ids[1]]], dim=1)


@torch.no_grad()
def evaluate(model, tokenizer, data, template, label_ids, device, batch_size=64):
    model.eval()
    correct, total_loss, n = 0, 0.0, 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        scores = forward_logits(model, tokenizer, [x["sentence"] for x in batch],
                                template, label_ids, device)
        labels = torch.tensor([x["label"] for x in batch], device=device)
        total_loss += F.cross_entropy(scores, labels, reduction="sum").item()
        correct += (scores.argmax(dim=1) == labels).sum().item()
        n += len(batch)
    return 100 * correct / n, total_loss / n


def reset_weights(model, init_state):
    """Remet les poids a leur etat initial (pre-entraine) -- en memoire, sans relire le disque."""
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.copy_(init_state[k])


def train_eval(model, tokenizer, template, label_ids, seed, lr, batch_size,
               epochs, device, eval_test, init_state):
    """Fine-tune le template sur (seed) et renvoie le meilleur (val_acc, val_loss, test_acc).
    eval_test=False -> test_acc reste None (gain de temps pendant le classement)."""
    reset_weights(model, init_state)
    torch.manual_seed(seed)
    train, val, test = load_sst2_fewshot(k=16, seed=seed)
    optimizer = AdamW(model.parameters(), lr=lr)

    best = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train)).tolist()
        for i in range(0, len(train), batch_size):
            batch = [train[j] for j in order[i:i + batch_size]]
            scores = forward_logits(model, tokenizer, [x["sentence"] for x in batch],
                                    template, label_ids, device)
            labels = torch.tensor([x["label"] for x in batch], device=device)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        val_acc, val_loss = evaluate(model, tokenizer, val, template, label_ids, device)
        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            test_acc = evaluate(model, tokenizer, test, template, label_ids, device)[0] if eval_test else None
            best = {"val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="templates.json")
    ap.add_argument("--rank_seed", type=int, default=42, help="seed pour le classement")
    ap.add_argument("--rank_lr", type=float, default=1e-5)
    ap.add_argument("--rank_bs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="selection_results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    label_ids = get_label_token_ids(tokenizer)
    # copie des poids initiaux (pre-entraines) pour reset rapide entre templates
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    with open(args.templates) as f:
        templates = json.load(f)
    print(f"{len(templates)} templates a classer (seed {args.rank_seed}, {args.epochs} epochs chacun)")

    # ----- ETAPE A : classement sur val (seed 42) -----
    ranked = []
    for i, t in enumerate(templates, 1):
        tmpl = t["template"]
        try:
            r = train_eval(model, tokenizer, tmpl, label_ids, args.rank_seed,
                           args.rank_lr, args.rank_bs, args.epochs, device,
                           eval_test=False, init_state=init_state)
        except Exception as e:
            print(f"  [{i:3d}/{len(templates)}] SKIP ({tmpl})  -> {e}")
            continue
        ranked.append({"template": tmpl, "variant": t.get("variant"),
                       "gen_score": t.get("score"),
                       "val_acc": r["val_acc"], "val_loss": round(r["val_loss"], 4)})
        print(f"  [{i:3d}/{len(templates)}] val {r['val_acc']:5.1f}% (loss {r['val_loss']:.3f})  {tmpl}")

    ranked.sort(key=lambda x: (x["val_acc"], -x["val_loss"]), reverse=True)
    with open(f"{args.out}.json", "w") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)
    with open(f"{args.out}.txt", "w") as f:
        for i, r in enumerate(ranked, 1):
            f.write(f"{i:3d}. val {r['val_acc']:5.1f}% (loss {r['val_loss']:.3f})  {r['template']}\n")

    print("\n===== TOP 10 templates (classement val, seed 42) =====")
    for r in ranked[:10]:
        print(f"  val {r['val_acc']:5.1f}% (loss {r['val_loss']:.3f})  {r['template']}")

    best_tmpl = ranked[0]["template"]
    print(f"\n>>> MEILLEUR template : {best_tmpl}")

    # ----- ETAPE B : evaluation finale du gagnant (5 seeds x grille) -----
    print("\n===== EVALUATION FINALE du meilleur template (5 seeds x grille) =====")
    test_scores = []
    for seed in SEEDS:
        best_cfg = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
        for (lr, bs) in GRID:
            r = train_eval(model, tokenizer, best_tmpl, label_ids, seed, lr, bs,
                           args.epochs, device, eval_test=True, init_state=init_state)
            if r["val_acc"] > best_cfg["val_acc"] or \
               (r["val_acc"] == best_cfg["val_acc"] and r["val_loss"] < best_cfg["val_loss"]):
                best_cfg = r
        print(f"  seed {seed} -> TEST {best_cfg['test_acc']:.1f}%")
        test_scores.append(best_cfg["test_acc"])

    mean, std = statistics.mean(test_scores), statistics.pstdev(test_scores)
    print("\n" + "=" * 50)
    print(f"Template auto-genere choisi : {best_tmpl}")
    print(f"Scores test par seed : {[round(s, 1) for s in test_scores]}")
    print(f"Prompt-based FT (AUTO template) : {mean:.1f} +/- {std:.1f}")
    print(f"(rappel) manuel, nous : 89.2 +/- 1.0   | papier auto : 92.3")
    print("=" * 50)

    # sauvegarde du resultat final
    with open("auto_template_result.json", "w") as f:
        json.dump({"best_template": best_tmpl, "test_scores": test_scores,
                   "mean": mean, "std": std}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
