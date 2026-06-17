"""
Sanity-check du prompt-based fine-tuning (LM-BFF, Gao et al. 2021).

Objectif : Voir le mécanisme du papier, SANS aucun entraînement (zero-shot).
On donne a RoBERTa une phrase reformulee avec le template "<phrase> It was [MASK] ."
et on regarde, entre les deux mots-labels "great" (positif) et "terrible" (negatif),
lequel il met sous [MASK]. C'est exactement l'equation (1) du papier.

Aucun poids n'est modifie ici : on lit juste la tete MLM deja pre-entrainee.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "roberta-large"

# --- Le prompt manuel (Table 1 du papier, ligne SST-2) ---
TEMPLATE = "{sentence} It was {mask} ."          # le template
LABEL_WORDS = {"positif": "great", "negatif": "terrible"}  # la label mapping M

# Quelques phrases de test (critiques de film) avec leur vraie classe
EXAMPLES = [
    ("a feel-good picture in the best sense of the term", "positif"),
    ("no reason to watch this mess", "negatif"),
    ("a masterpiece, beautifully acted and directed", "positif"),
    ("comes off as a loud, lurid mess", "negatif"),
    ("one of the greatest films ever made", "positif"),
    ("boring, predictable and a complete waste of time", "negatif"),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Chargement de {MODEL_NAME} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()

    mask_token = tokenizer.mask_token          # "<mask>" pour RoBERTa

    # Les identifiants (token ids) des deux mots-labels dans le vocabulaire.
    # RoBERTa encode un mot precede d'un espace avec un prefixe special ; on
    # recupere donc l'id du token " great" / " terrible".
    label_ids = {
        cls: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + word)[0])
        for cls, word in LABEL_WORDS.items()
    }
    print("Mots-labels -> token ids :", label_ids)
    print("=" * 70)

    correct = 0
    for sentence, gold in EXAMPLES:
        # 1. Construire l'entree "promptee"
        text = TEMPLATE.format(sentence=sentence, mask=mask_token)
        enc = tokenizer(text, return_tensors="pt").to(device)

        # 2. Forward (pas de gradient : on ne fait que lire)
        with torch.no_grad():
            logits = model(**enc).logits        # [1, longueur, taille_vocab]

        # 3. Trouver la position du [MASK] et lire ses scores
        mask_pos = (enc.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        mask_logits = logits[0, mask_pos[0]]     # [taille_vocab]

        # 4. Ne garder QUE les scores de "great" et "terrible", softmax sur les 2
        scores = torch.tensor([mask_logits[label_ids["positif"]],
                               mask_logits[label_ids["negatif"]]])
        probs = torch.softmax(scores, dim=0)
        p_pos, p_neg = probs[0].item(), probs[1].item()

        pred = "positif" if p_pos > p_neg else "negatif"
        ok = (pred == gold)
        correct += ok

        print(f"Phrase  : {sentence}")
        print(f"  P(great|positif)={p_pos:.3f}  P(terrible|negatif)={p_neg:.3f}"
              f"  -> pred={pred}  (vrai={gold})  {'OK' if ok else 'X'}")
        print("-" * 70)

    print(f"\nAccuracy zero-shot (sans aucun entrainement) : {correct}/{len(EXAMPLES)}")
    print("=> Si c'est deja eleve, c'est la preuve que le prompt reutilise")
    print("   les connaissances du pre-entrainement. Le fine-tuning viendra ameliorer ca.")


if __name__ == "__main__":
    main()
