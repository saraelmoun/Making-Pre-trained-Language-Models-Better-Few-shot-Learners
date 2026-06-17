"""
Auto-generation des TEMPLATES avec T5 (LM-BFF, Gao et al. 2021, section 5.2).

FIDELE AU PAPIER :
  - Modele : T5-3B
  - Entree : phrase a trous avec le VRAI mot-label insere (pas [MASK]).
    Pour une tache a PHRASE UNIQUE (SST-2), le papier (note 7) considere DEUX variantes :
        variante "apres" :  {phrase} <X> M(y) <Y>
        variante "avant" :  <X> M(y) <Y> {phrase}
  - Objectif (eq. 4) : trouver le template qui maximise la log-proba sur TOUS les
    exemples a la fois (beam search JOINT).
  - Beam search large (~100) -> plusieurs candidats diversifies.
  - FUSION : on reunit les candidats des deux variantes en un seul pool, sans doublons.

MEMOIRE (pas une divergence) : le beam search joint traite beam x N_exemples sequences.
Pour ne pas saturer la GPU, on decoupe ces sequences en PAQUETS (--chunk beams a la fois)
et on additionne -> maths identiques, juste calculees en morceaux.

Sortie : templates.json (structure) + templates.txt (lisible). Aucune selection ici.
T5 utilise les sentinelles <extra_id_0>, <extra_id_1>, ... a la place de <X>, <Y>.
"""

import argparse
import json
import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from data import load_sst2_fewshot

# evite la fragmentation memoire (beam search = bcp d'allocations)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LABEL_WORDS = {1: "great", 0: "terrible"}   # 1=positif, 0=negatif


def build_t5_inputs(train, tokenizer, device, variant):
    """Une entree T5 par exemple, selon la variante.

    variant='after' :  '{phrase} <extra_id_0> {mot-label} <extra_id_1>'
    variant='before':  '<extra_id_0> {mot-label} <extra_id_1> {phrase}'
    """
    if variant == "after":
        texts = [f"{x['sentence']} <extra_id_0> {LABEL_WORDS[x['label']]} <extra_id_1>"
                 for x in train]
    else:  # before
        texts = [f"<extra_id_0> {LABEL_WORDS[x['label']]} <extra_id_1> {x['sentence']}"
                 for x in train]
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)


@torch.no_grad()
def joint_beam_search(model, tokenizer, enc, beam=100, max_len=20, n_return=100, chunk=16):
    """Beam search JOINT : score d'une sequence = moyenne des log-proba sur tous les
    exemples (eq. 4). Le forward sur (beam x N) sequences est DECOUPE en paquets de
    'chunk' beams pour borner la memoire (resultat identique). Renvoie [(texte, score)].
    """
    device = enc.input_ids.device
    N = enc.input_ids.size(0)
    V = model.config.vocab_size
    eos_id = tokenizer.eos_token_id

    # Encoder : on encode les N exemples une seule fois
    encoder_outputs = model.get_encoder()(input_ids=enc.input_ids,
                                           attention_mask=enc.attention_mask)
    enc_hidden = encoder_outputs.last_hidden_state  # [N, L, d]
    enc_mask = enc.attention_mask                   # [N, L]
    L, d = enc_hidden.shape[1], enc_hidden.shape[2]

    start_id = model.config.decoder_start_token_id
    beams = [([start_id], 0.0)]
    completed = []

    for step in range(max_len):
        if not beams:
            break
        B = len(beams)
        t = len(beams[0][0])
        dec = torch.tensor([b[0] for b in beams], device=device)  # [B, t]

        # --- calcul du score joint [B, V], DECOUPE en paquets de 'chunk' beams ---
        joint = torch.empty(B, V, device=device)
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            g = e - s                                              # nb de beams du paquet
            dec_g = dec[s:e].unsqueeze(1).expand(g, N, t).reshape(g * N, t)   # [g*N, t]
            hid = enc_hidden.unsqueeze(0).expand(g, N, L, d).reshape(g * N, L, d)
            msk = enc_mask.unsqueeze(0).expand(g, N, L).reshape(g * N, L)
            out = model(encoder_outputs=(hid,), attention_mask=msk, decoder_input_ids=dec_g)
            logp = torch.log_softmax(out.logits[:, -1, :], dim=-1).view(g, N, V)
            joint[s:e] = logp.mean(dim=1)                          # moyenne sur les exemples

        # --- expansion beam search ---
        total = torch.tensor([b[1] for b in beams], device=device).unsqueeze(1) + joint  # [B, V]
        flat = total.view(-1)
        topv, topi = flat.topk(min(beam * 2, flat.numel()))

        new_beams = []
        for val, idx in zip(topv.tolist(), topi.tolist()):
            b_idx, tok = divmod(idx, V)
            seq = beams[b_idx][0] + [tok]
            if tok == eos_id:
                completed.append((seq, val / max(len(seq) - 1, 1)))   # normalise par longueur
            else:
                new_beams.append((seq, val))
            if len(new_beams) >= beam:
                break
        beams = new_beams

    for seq, sc in beams:
        completed.append((seq, sc / max(len(seq) - 1, 1)))

    completed.sort(key=lambda x: x[1], reverse=True)
    results, seen = [], set()
    for seq, sc in completed:
        text = tokenizer.decode(seq, skip_special_tokens=False)
        if text in seen:
            continue
        seen.add(text)
        results.append((text, sc))
        if len(results) >= n_return:
            break
    return results


def to_template(decoded_text, variant):
    """Reconstruit le template a partir de la sortie T5 '<extra_id_0> A <extra_id_1> B'.

    variant='after'  -> '{sentence} A {mask} B'
    variant='before' -> 'A {mask} B {sentence}'
    Renvoie None si format inexploitable.
    """
    s0, s1, s2 = "<extra_id_0>", "<extra_id_1>", "<extra_id_2>"
    if s0 not in decoded_text or s1 not in decoded_text:
        return None
    after0 = decoded_text.split(s0, 1)[1]
    textA, after1 = after0.split(s1, 1) if s1 in after0 else (after0, "")
    textB = after1
    for stop in (s2, "</s>", "<pad>"):
        if stop in textB:
            textB = textB.split(stop, 1)[0]
    textA, textB = textA.strip(), textB.strip()
    if variant == "after":
        tmpl = f"{{sentence}} {textA} {{mask}} {textB}"
    else:
        tmpl = f"{textA} {{mask}} {textB} {{sentence}}"
    return " ".join(tmpl.split())   # normalise les espaces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="t5-3b", help="t5-3b = celui du papier (defaut)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--n_return", type=int, default=100, help="candidats par variante")
    ap.add_argument("--chunk", type=int, default=16, help="beams par paquet (memoire)")
    ap.add_argument("--out", default="templates")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | modele : {args.model} | beam : {args.beam} | chunk : {args.chunk}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device).eval()

    train, _, _ = load_sst2_fewshot(k=16, seed=args.seed)
    print(f"Generation a partir de {len(train)} exemples (seed {args.seed})")

    # --- les DEUX variantes, puis FUSION ---
    templates, seen = [], set()
    for variant in ("after", "before"):
        print(f"\n--- variante '{variant}' ---")
        enc = build_t5_inputs(train, tokenizer, device, variant)
        raw = joint_beam_search(model, tokenizer, enc, beam=args.beam,
                                max_len=args.max_len, n_return=args.n_return, chunk=args.chunk)
        added = 0
        for text, score in raw:
            tmpl = to_template(text, variant)
            if tmpl and tmpl not in seen:
                seen.add(tmpl)
                templates.append({"template": tmpl, "score": round(score, 4), "variant": variant})
                added += 1
        print(f"  {added} nouveaux templates")

    # tri global par score
    templates.sort(key=lambda x: x["score"], reverse=True)

    with open(f"{args.out}.json", "w") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)
    with open(f"{args.out}.txt", "w") as f:
        for i, t in enumerate(templates, 1):
            f.write(f"{i:3d}. [{t['variant']:6s}] (score {t['score']:+.3f})  {t['template']}\n")

    print(f"\n{len(templates)} templates au total (apres fusion) -> {args.out}.json / {args.out}.txt")
    print("Apercu des 15 premiers :")
    for t in templates[:15]:
        print(f"   [{t['variant']:6s}] (score {t['score']:+.3f})  {t['template']}")
    print(f"\n(rappel) template manuel du papier : '{{sentence}} It was {{mask}} .'")


if __name__ == "__main__":
    main()
