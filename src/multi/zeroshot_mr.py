"""
ETAPE 3 — Zero-shot prompt-based sur MR (fichier NEUF, isole).

On applique le prompt manuel (template + mots-labels) sur les 1066 phrases de test,
SANS aucun entrainement, et on lit la tete MLM de RoBERTa. Meme mecanique que pour
SST-2, mais sur MR. Sauvegarde 03_zeroshot.json / .txt.
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data_mr import load_mr_fewshot

MODEL_NAME = "roberta-large"
TEMPLATE = "{sentence} It was {mask} ."
LABEL_WORDS = {1: "great", 0: "terrible"}     # 1=positif, 0=negatif (Table 1 du papier)
BATCH_SIZE = 32
OUT_DIR = "outputs_mr"


def get_label_token_ids(tokenizer):
    return {cls: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + w)[0])
            for cls, w in LABEL_WORDS.items()}


@torch.no_grad()
def predict_batch(model, tokenizer, sentences, label_ids, device):
    texts = [TEMPLATE.format(sentence=s, mask=tokenizer.mask_token) for s in sentences]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(**enc).logits
    mask_logits = logits[enc.input_ids == tokenizer.mask_token_id]      # [B, vocab]
    two = torch.stack([mask_logits[:, label_ids[0]], mask_logits[:, label_ids[1]]], dim=1)
    return two.argmax(dim=1).tolist()                                  # 0=neg, 1=pos


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | modele : {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    label_ids = get_label_token_ids(tokenizer)

    _, _, test = load_mr_fewshot(k=16, seed=42)
    print(f"Test set MR : {len(test)} phrases")

    correct = 0
    for i in range(0, len(test), BATCH_SIZE):
        batch = test[i:i + BATCH_SIZE]
        preds = predict_batch(model, tokenizer, [x["sentence"] for x in batch], label_ids, device)
        correct += sum(p == x["label"] for p, x in zip(preds, batch))
    acc = 100 * correct / len(test)

    report = {
        "step": "03_zeroshot",
        "dataset": "MR (rotten_tomatoes)",
        "method": "prompt-based zero-shot (manuel)",
        "template": TEMPLATE,
        "label_words": LABEL_WORDS,
        "n_test": len(test),
        "accuracy": round(acc, 1),
        "correct": correct,
        "note": "aucun entrainement ; on lit la tete MLM pre-entrainee",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "03_zeroshot.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "03_zeroshot.txt"), "w") as f:
        f.write("ETAPE 3 — ZERO-SHOT prompt-based sur MR\n")
        f.write("=" * 40 + "\n")
        f.write(f"Template     : {TEMPLATE}\n")
        f.write(f"Mots-labels  : {LABEL_WORDS}\n")
        f.write(f"Test set     : {len(test)} phrases\n\n")
        f.write(f"Accuracy ZERO-SHOT : {acc:.1f}%  ({correct}/{len(test)})\n\n")
        f.write("Aucun entrainement : le prompt reutilise les connaissances du pre-entrainement.\n")

    print(f"\nAccuracy ZERO-SHOT sur MR : {acc:.1f}%  ({correct}/{len(test)})")
    print(f"Rapport -> {OUT_DIR}/03_zeroshot.json et .txt")


if __name__ == "__main__":
    main()
