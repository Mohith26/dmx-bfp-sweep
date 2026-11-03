"""Perplexity of OPT-125m on WikiText-2 under dmx-compressor numerical formats."""
import argparse, json, math, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dmx.compressor import config_rules, format as F
from dmx.compressor.modeling import DmxModel, DmxConfigRule, nn as dnn

MODEL = "facebook/opt-125m"


def linear_rule(fmt):
    """Cast Linear inputs and weights to `fmt`, everything else left as-is."""
    return [DmxConfigRule(module_types=(dnn.Linear,), module_config=dict(
        input_formats=[fmt], weight_format=fmt))]


def weight_only_rule(fmt):
    return [DmxConfigRule(module_types=(dnn.Linear,), module_config=dict(weight_format=fmt))]


CONFIGS = {
    "baseline": None,
    "BASIC": list(config_rules.BASIC),
}
for name in ["BFP16_64", "BFP14_64", "BFP12_128", "BFP12_64", "BFP12_32", "BFP12_16"]:
    CONFIGS[f"linear_in+w_{name}"] = linear_rule(getattr(F, name))
for name in ["BFP12_64"]:
    CONFIGS[f"linear_w_only_{name}"] = weight_only_rule(getattr(F, name))


def windows(tok, n_windows, seqlen):
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0) for i in range(n_windows)]


@torch.no_grad()
def perplexity(model, wins):
    nll, count = 0.0, 0
    for w in wins:
        out = model(input_ids=w, labels=w)
        n = w.shape[1] - 1
        nll += out.loss.item() * n
        count += n
    return math.exp(nll / count)


def build(rules):
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    if rules is None:
        return m
    dm = DmxModel.from_torch(m)
    return dm.transform(dm.dmx_config, *rules)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    wins = windows(tok, a.windows, a.seqlen)
    results = {}
    for name, rules in CONFIGS.items():
        if a.only and name not in a.only:
            continue
        t = time.time()
        ppl = perplexity(build(rules), wins)
        results[name] = {"ppl": round(ppl, 3), "seconds": round(time.time() - t, 1)}
        print(f"{name:28s} ppl={ppl:8.3f}  ({results[name]['seconds']}s)", flush=True)
    json.dump({"model": MODEL, "windows": a.windows, "seqlen": a.seqlen,
               "tokens_scored": a.windows * (a.seqlen - 1), "results": results},
              open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
