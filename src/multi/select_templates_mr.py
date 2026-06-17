"""
ETAPE 8b — SELECTION des templates auto-generes (§5.2) sur MR (fichier NEUF, independant).

Lit 08a_templates.json, fine-tune chaque candidat sur seed 42 (classement sur val),
puis evalue le gagnant sur 5 seeds. Cible papier (Table 3, MR, auto) : 88.5.
Sauvegarde 08b_template_selection.json + 08_auto_template.json/.txt.
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
LABEL_WORDS = {1: "great", 0: "terrible"}
SEEDS = [13, 21, 42, 87, 100]
GRID = [(1e-5, 2), (1e-5, 4), (2e-5, 4), (2e-5, 8)]
EPOCHS = 60
RANK_SEED = 42
PAPER_TARGET = 88.5
OUT_DIR = "outputs_mr"


def get_label_token_ids(tokenizer):
    return {cls: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + w)[0])
            for cls, w in LABEL_WORDS.items()}


def forward_logits(model, tokenizer, sentences, template, label_ids, device):
    texts = [template.format(sentence=s, mask=tokenizer.mask_token) for s in sentences]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(**enc).logits
    mask_logits = logits[enc.input_ids == tokenizer.mask_token_id]
    return torch.stack([mask_logits[:, label_ids[0]], mask_logits[:, label_ids[1]]], dim=1)


@torch.no_grad()
def evaluate(model, tokenizer, data, template, label_ids, device, batch_size=64):
    model.eval()
    correct, total_loss, n = 0, 0.0, 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        scores = forward_logits(model, tokenizer, [x["sentence"] for x in batch], template, label_ids, device)
        labels = torch.tensor([x["label"] for x in batch], device=device)
        total_loss += F.cross_entropy(scores, labels, reduction="sum").item()
        correct += (scores.argmax(dim=1) == labels).sum().item()
        n += len(batch)
    return 100 * correct / n, total_loss / n


def reset_weights(model, init_state):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.copy_(init_state[k])


def train_eval(model, tokenizer, template, label_ids, seed, lr, bs, epochs, device, eval_test, init_state):
    reset_weights(model, init_state)
    torch.manual_seed(seed)
    train, val, test = load_mr_fewshot(k=16, seed=seed)
    optimizer = AdamW(model.parameters(), lr=lr)
    best = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train)).tolist()
        for i in range(0, len(train), bs):
            batch = [train[j] for j in order[i:i + bs]]
            scores = forward_logits(model, tokenizer, [x["sentence"] for x in batch], template, label_ids, device)
            labels = torch.tensor([x["label"] for x in batch], device=device)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        val_acc, val_loss = evaluate(model, tokenizer, val, template, label_ids, device)
        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            test_acc = evaluate(model, tokenizer, test, template, label_ids, device)[0] if eval_test else None
            best = {"val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc}
    return best


def save(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    label_ids = get_label_token_ids(tokenizer)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    with open(os.path.join(OUT_DIR, "08a_templates.json")) as f:
        templates = json.load(f)
    print(f"{len(templates)} templates MR a classer (seed {RANK_SEED}, {EPOCHS} epochs)")

    # ---- classement sur val ----
    ranked = []
    for i, t in enumerate(templates, 1):
        tmpl = t["template"]
        try:
            r = train_eval(model, tokenizer, tmpl, label_ids, RANK_SEED, 1e-5, 4, EPOCHS, device, False, init_state)
        except Exception as e:
            print(f"  [{i:3d}] SKIP ({tmpl}) -> {e}"); continue
        ranked.append({"template": tmpl, "variant": t.get("variant"),
                       "val_acc": r["val_acc"], "val_loss": round(r["val_loss"], 4)})
        print(f"  [{i:3d}/{len(templates)}] val {r['val_acc']:5.1f}% (loss {r['val_loss']:.3f})  {tmpl}")
    ranked.sort(key=lambda x: (x["val_acc"], -x["val_loss"]), reverse=True)
    save("08b_template_selection.json", ranked)
    best_tmpl = ranked[0]["template"]
    print(f"\n>>> MEILLEUR template MR : {best_tmpl}")

    # ---- eval finale 5 seeds ----
    print("\nEvaluation finale (5 seeds x grille)...")
    test_scores = []
    for seed in SEEDS:
        bc = {"val_acc": -1, "val_loss": float("inf"), "test_acc": None}
        for (lr, bs) in GRID:
            r = train_eval(model, tokenizer, best_tmpl, label_ids, seed, lr, bs, EPOCHS, device, True, init_state)
            if r["val_acc"] > bc["val_acc"] or (r["val_acc"] == bc["val_acc"] and r["val_loss"] < bc["val_loss"]):
                bc = r
        print(f"  seed {seed} -> TEST {bc['test_acc']:.1f}%")
        test_scores.append(bc["test_acc"])
    mean, std = statistics.mean(test_scores), statistics.pstdev(test_scores)

    result = {"step": "08_auto_template", "dataset": "MR (rotten_tomatoes)",
              "best_template": best_tmpl, "test_scores": [round(s, 1) for s in test_scores],
              "mean": round(mean, 1), "std": round(std, 1), "paper_target": PAPER_TARGET,
              "note_manuel_nous": "It was [MASK] -> 85.4 +/- 2.7"}
    save("08_auto_template.json", result)
    with open(os.path.join(OUT_DIR, "08_auto_template.txt"), "w") as f:
        f.write("ETAPE 8 — AUTO TEMPLATE (§5.2) sur MR\n" + "=" * 45 + "\n")
        f.write(f"Template auto choisi : {best_tmpl}\n")
        f.write(f"Scores test : {[round(s,1) for s in test_scores]}\n")
        f.write(f"Prompt-based FT (AUTO template) : {mean:.1f} +/- {std:.1f}\n")
        f.write(f"Cible papier (Table 3, MR auto) : {PAPER_TARGET}\n")
        f.write(f"(rappel) manuel It was [MASK], nous : 85.4 +/- 2.7\n")

    print("\n" + "=" * 50)
    print(f"Template auto MR : {best_tmpl}")
    print(f"Prompt-based FT (AUTO template) : {mean:.1f} +/- {std:.1f}")
    print(f"Cible papier : {PAPER_TARGET} | manuel nous : 85.4 +/- 2.7")
    print("=" * 50)


if __name__ == "__main__":
    main()
