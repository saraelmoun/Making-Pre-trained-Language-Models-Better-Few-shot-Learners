"""
Reproduction zero-shot de Min et al. (2022), Noisy Channel — version batchée.

3 scorings (direct / direct++ / channel) x 4 templates (Table 6) x 2 datasets
(SST-2, Subj : les deux datasets communs aux deux papiers).
Test sets COMPLETS. On reporte moyenne / pire-cas / ecart-type, + comparaison Table 3.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL = "gpt2-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()

# --- datasets (nyu-mll/glue car "glue" seul casse avec datasets recents) ---
DATASETS = {
    "sst2": {
        "load": lambda: load_dataset("nyu-mll/glue", "sst2")["validation"],
        "text_key": "sentence", "label_key": "label", "labels": [0, 1],
        "verbalizers": {0: "terrible", 1: "great"},
        "templates": ["A MASK one.", "It was MASK.", "All in all MASK.", "A MASK piece."],
    },
    "subj": {
        "load": lambda: load_dataset("SetFit/subj")["test"],
        "text_key": "text", "label_key": "label", "labels": [0, 1],
        "verbalizers": {0: "objective", 1: "subjective"},
        "templates": ["This is MASK.", "It's all MASK.", "It's MASK.", "Is it MASK?"],
    },
}

PAPER_TABLE3 = {
    "sst2": {"direct": (63.0, 51.1), "direct++": (80.3, 76.9), "channel": (77.1, 74.8)},
    "subj": {"direct": (51.0, 49.9), "direct++": (52.0, 48.8), "channel": (57.8, 51.5)},
}


def split_template(t):
    left, right = t.split("MASK")
    return left, right


@torch.no_grad()
def batched_logprob(pairs, batch_size=64, length_normalize=True):
    scores = [0.0] * len(pairs)
    for s in range(0, len(pairs), batch_size):
        chunk = pairs[s:s + batch_size]
        prefix_ids = [tokenizer(p, add_special_tokens=False).input_ids for p, c in chunk]
        full_ids = [tokenizer(p + c, add_special_tokens=False).input_ids for p, c in chunk]
        n_prefix = [len(x) for x in prefix_ids]
        n_full = [len(x) for x in full_ids]
        maxlen = max(n_full)
        input_ids = torch.full((len(chunk), maxlen), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, ids in enumerate(full_ids):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attn[i, :len(ids)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        logp = F.log_softmax(model(input_ids=input_ids, attention_mask=attn).logits, dim=-1)
        for i in range(len(chunk)):
            np_, nf_ = n_prefix[i], n_full[i]
            nc = nf_ - np_
            if nc <= 0:
                scores[s + i] = float("-inf"); continue
            pos = torch.arange(np_ - 1, nf_ - 1, device=device)
            tok = input_ids[i, np_:nf_]
            lp = logp[i, pos, :].gather(1, tok.unsqueeze(1)).squeeze(1).sum().item()
            scores[s + i] = lp / nc if length_normalize else lp
    return scores


def make_pairs(method, x, word, template):
    left, right = split_template(template)
    x = x.strip()
    if method in ("direct", "direct++"):
        pair_in = ((x + " " + left).rstrip(), " " + word)
        if method == "direct":
            return [pair_in]
        return [pair_in, (left.rstrip(), " " + word)]
    return [(left + word + right, " " + x)]


def evaluate_template(name, method, template, batch_size=64):
    cfg = DATASETS[name]; words = cfg["verbalizers"]; labels = cfg["labels"]
    data = cfg["load"]()
    texts = [ex[cfg["text_key"]] for ex in data]
    gold = [ex[cfg["label_key"]] for ex in data]
    all_pairs, index = [], []
    for i, x in enumerate(texts):
        for lab in labels:
            for k, pr in enumerate(make_pairs(method, x, words[lab], template)):
                index.append((i, lab, k)); all_pairs.append(pr)
    flat = batched_logprob(all_pairs, batch_size=batch_size)
    score_by = {}
    for (i, lab, k), sc in zip(index, flat):
        if method == "direct++":
            score_by[(i, lab)] = score_by.get((i, lab), 0.0) + (sc if k == 0 else -sc)
        else:
            score_by[(i, lab)] = sc
    correct = sum(labels[int(np.argmax([score_by[(i, l)] for l in labels]))] == gold[i]
                  for i in range(len(texts)))
    return correct / len(texts)


def main():
    print(f"Modele {MODEL} | device {device}")
    results = {}
    for name in ["sst2", "subj"]:
        results[name] = {}
        for method in ["direct", "direct++", "channel"]:
            accs = []
            for t, tmpl in enumerate(DATASETS[name]["templates"]):
                a = evaluate_template(name, method, tmpl)
                accs.append(a)
                print(f"{name:5s} {method:9s} t{t+1} ({tmpl:15s}) -> {a*100:5.1f}%")
            accs = np.array(accs)
            results[name][method] = (accs.mean()*100, accs.min()*100, accs.std()*100)
            print(f"   => {name}/{method}: moy {accs.mean()*100:.1f} "
                  f"pire {accs.min()*100:.1f} std {accs.std()*100:.1f}\n")

    print("\n===== SYNTHESE (moyenne / pire-cas ; std) vs PAPIER =====")
    print(f"{'dataset':6s} {'methode':9s} | {'nous moy/pire (std)':22s} | {'papier moy/pire':16s}")
    for name in ["sst2", "subj"]:
        for method in ["direct", "direct++", "channel"]:
            mo, wo, sd = results[name][method]
            pm, pw = PAPER_TABLE3[name][method]
            print(f"{name:6s} {method:9s} | {mo:5.1f}/{wo:5.1f} ({sd:4.1f})        | {pm:5.1f}/{pw:5.1f}")


if __name__ == "__main__":
    main()
