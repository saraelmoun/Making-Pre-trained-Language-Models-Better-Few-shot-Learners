# Reproduction LM-BFF (Gao et al. 2021) — SST-2

Modèle : RoBERTa-large | K=16 (16 ex./classe) | 5 seeds [13,21,42,87,100]
Protocole : grille hyperparamètres (lr × batch), sélection sur val (départage par perte val),
moyenne ± écart-type sur 5 seeds. Test = split validation officiel SST-2 (872 phrases).

## Tableau de résultats (notre reproduction de la Table 3)

| Méthode                       | Nous          | Papier      | Script             |
|-------------------------------|---------------|-------------|--------------------|
| Prompt-based zero-shot        | 81.7          | 83.6        | zero_shot_eval.py  |
| Fine-tuning classique         | 75.5 ± 6.3    | 81.4 ± 3.8  | baseline.py        |
| Prompt-based FT (manuel)      | 89.2 ± 1.0    | 92.7 ± 0.9  | train.py           |
| Prompt-based FT (auto T5)     | — (à faire)   | 92.3        | (à venir, §5.2)    |

## Détail par seed

Prompt-based FT (manuel) : [87.8, 90.3, 90.6, 88.9, 88.5] -> 89.2 ± 1.0
Fine-tuning classique    : [74.5, 67.9, 69.6, 81.3, 84.1] -> 75.5 ± 6.3

## Conclusions (thèse du papier, reproduite)

1. Prompt-based >> classique : +13.7 points (89.2 vs 75.5) avec les mêmes 32 exemples.
2. Prompt-based bien plus stable : écart-type 1.0 vs 6.3 (~6x). Le classique va de 67% à 84%
   selon le tirage (une config tombe même à 49% = hasard).
3. Le zero-shot (81.7%) montre que le prompt exploite déjà le pré-entraînement sans entraînement.
4. Écart absolu ~3 pts sous le papier (grille d'hyperparamètres réduite), mais tendance identique.

## Template manuel utilisé (Table 1 du papier)

Template    : "{sentence} It was [MASK] ."
Mots-labels : positif -> great, négatif -> terrible

## Reste à faire (partie 1 = prompt-based FT + pipeline auto)

- [ ] §5.2 Auto-génération des templates via T5 (le "pipeline" — cœur restant)
- [ ] (optionnel) §5.1 Auto-sélection des mots-labels
- [ ] (optionnel) élargir à une 2e tâche (SST-5 / TREC / SNLI)
