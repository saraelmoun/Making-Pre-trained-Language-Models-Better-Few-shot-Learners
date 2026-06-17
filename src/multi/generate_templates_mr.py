"""
ETAPE 8a — Auto-generation des TEMPLATES avec T5 (§5.2) sur MR (fichier NEUF, independant).

Identique a la version SST-2 (T5-3B, beam joint eq.4, 2 variantes + fusion, decoupage memoire),
mais pointe sur data_mr. Mots-labels MR = {1:great, 0:terrible} (memes que SST-2, Table 1).
Sauvegarde 08a_templates.json / .txt.
"""

import argparse
import json
import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from data_mr import load_mr_fewshot

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
LABEL_WORDS = {1: "great", 0: "terrible"}
OUT_DIR = "outputs_mr"


def build_t5_inputs(train, tokenizer, device, variant):
    if variant == "after":
        texts = [f"{x['sentence']} <extra_id_0> {LABEL_WORDS[x['label']]} <extra_id_1>" for x in train]
    else:
        texts = [f"<extra_id_0> {LABEL_WORDS[x['label']]} <extra_id_1> {x['sentence']}" for x in train]
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)


@torch.no_grad()
def joint_beam_search(model, tokenizer, enc, beam=100, max_len=20, n_return=100, chunk=16):
    device = enc.input_ids.device
    N = enc.input_ids.size(0)
    V = model.config.vocab_size
    eos_id = tokenizer.eos_token_id
    encoder_outputs = model.get_encoder()(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    enc_hidden = encoder_outputs.last_hidden_state
    enc_mask = enc.attention_mask
    L, d = enc_hidden.shape[1], enc_hidden.shape[2]
    start_id = model.config.decoder_start_token_id
    beams = [([start_id], 0.0)]
    completed = []
    for _ in range(max_len):
        if not beams:
            break
        B = len(beams)
        t = len(beams[0][0])
        dec = torch.tensor([b[0] for b in beams], device=device)
        joint = torch.empty(B, V, device=device)
        for s in range(0, B, chunk):
            e = min(s + chunk, B); g = e - s
            dec_g = dec[s:e].unsqueeze(1).expand(g, N, t).reshape(g * N, t)
            hid = enc_hidden.unsqueeze(0).expand(g, N, L, d).reshape(g * N, L, d)
            msk = enc_mask.unsqueeze(0).expand(g, N, L).reshape(g * N, L)
            out = model(encoder_outputs=(hid,), attention_mask=msk, decoder_input_ids=dec_g)
            logp = torch.log_softmax(out.logits[:, -1, :], dim=-1).view(g, N, V)
            joint[s:e] = logp.mean(dim=1)
        total = torch.tensor([b[1] for b in beams], device=device).unsqueeze(1) + joint
        topv, topi = total.view(-1).topk(min(beam * 2, total.numel()))
        new_beams = []
        for val, idx in zip(topv.tolist(), topi.tolist()):
            b_idx, tok = divmod(idx, V)
            seq = beams[b_idx][0] + [tok]
            if tok == eos_id:
                completed.append((seq, val / max(len(seq) - 1, 1)))
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
        seen.add(text); results.append((text, sc))
        if len(results) >= n_return:
            break
    return results


def to_template(decoded_text, variant):
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
    tmpl = f"{{sentence}} {textA} {{mask}} {textB}" if variant == "after" else f"{textA} {{mask}} {textB} {{sentence}}"
    return " ".join(tmpl.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="t5-3b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--n_return", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | modele : {args.model} | beam : {args.beam} (MR)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    train, _, _ = load_mr_fewshot(k=16, seed=args.seed)
    print(f"Generation a partir de {len(train)} exemples MR (seed {args.seed})")

    templates, seen = [], set()
    for variant in ("after", "before"):
        print(f"\n--- variante '{variant}' ---")
        enc = build_t5_inputs(train, tokenizer, device, variant)
        raw = joint_beam_search(model, tokenizer, enc, beam=args.beam, n_return=args.n_return, chunk=args.chunk)
        added = 0
        for text, score in raw:
            tmpl = to_template(text, variant)
            if tmpl and tmpl not in seen:
                seen.add(tmpl)
                templates.append({"template": tmpl, "score": round(score, 4), "variant": variant})
                added += 1
        print(f"  {added} nouveaux templates")
    templates.sort(key=lambda x: x["score"], reverse=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "08a_templates.json"), "w") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "08a_templates.txt"), "w") as f:
        for i, t in enumerate(templates, 1):
            f.write(f"{i:3d}. [{t['variant']:6s}] (score {t['score']:+.3f})  {t['template']}\n")

    print(f"\n{len(templates)} templates (MR) -> {OUT_DIR}/08a_templates.json / .txt")
    for t in templates[:12]:
        print(f"   [{t['variant']:6s}] (score {t['score']:+.3f})  {t['template']}")


if __name__ == "__main__":
    main()
