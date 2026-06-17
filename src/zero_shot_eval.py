"""
Evaluation ZERO-SHOT du prompt-based prediction sur TOUT le test set SST-2.

Aucun entrainement : on applique le template manuel + mots-labels et on lit la
tete MLM, exactement comme sanity_check.py, mais cette fois sur les ~872 phrases
reelles du test (et non 6 phrases triees). On vise le chiffre du papier : ~83.6.

C'est la ligne "Prompt-based zero-shot" de la Table 3.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data import load_sst2_fewshot, LABELS

MODEL_NAME = "roberta-large"
TEMPLATE = "{sentence} It was {mask} ."
# label mapping M : classe (int) -> mot-label
LABEL_WORDS = {1: "great", 0: "terrible"}   # 1=positif, 0=negatif
BATCH_SIZE = 32


def get_label_token_ids(tokenizer):
    """token id du mot-label, en tenant compte du prefixe d'espace de RoBERTa."""
    return {cls: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + w)[0])
            for cls, w in LABEL_WORDS.items()}


@torch.no_grad()
def predict_batch(model, tokenizer, sentences, label_ids, device):
    texts = [TEMPLATE.format(sentence=s, mask=tokenizer.mask_token) for s in sentences]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(**enc).logits                     # [B, L, V]

    # position du [MASK] dans chaque phrase
    mask_pos = (enc.input_ids == tokenizer.mask_token_id)
    # logits a la position [MASK] : [B, V]
    mask_logits = logits[mask_pos]

    # ne garder que les colonnes des mots-labels, softmax -> proba par classe
    pos_id, neg_id = label_ids[1], label_ids[0]
    two = torch.stack([mask_logits[:, neg_id], mask_logits[:, pos_id]], dim=1)  # [B,2] (neg,pos)
    preds = two.argmax(dim=1)                          # 0=neg, 1=pos
    return preds.tolist()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | modele : {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    label_ids = get_label_token_ids(tokenizer)
    print("Mots-labels -> token ids :", label_ids)

    _, _, test = load_sst2_fewshot(k=16, seed=42)
    print(f"Test set : {len(test)} phrases")

    correct = 0
    for i in range(0, len(test), BATCH_SIZE):
        batch = test[i:i + BATCH_SIZE]
        preds = predict_batch(model, tokenizer,
                              [x["sentence"] for x in batch], label_ids, device)
        for x, p in zip(batch, preds):
            correct += (p == x["label"])

    acc = 100 * correct / len(test)
    print(f"\nAccuracy ZERO-SHOT sur tout SST-2 : {acc:.1f}%  ({correct}/{len(test)})")
    print("Chiffre attendu du papier (Table 3) : ~83.6%")


if __name__ == "__main__":
    main()
