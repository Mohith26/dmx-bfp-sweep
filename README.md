# dmx-bfp-sweep

How much perplexity does OPT-125m lose under d-Matrix's block floating point (BFP) formats? I used [dmx-compressor](https://github.com/d-matrix-ai/dmx-compressor) to cast the model's Linear layers to different BFP formats and measured WikiText-2 perplexity on CPU.

## Results

`facebook/opt-125m`, first 16 non-overlapping 512-token windows of the WikiText-2 (raw) test split, 8,176 scored tokens. Numbers are only comparable to each other, not to published full-test-set perplexities.

| Config | What is cast | Perplexity | vs. baseline |
|---|---|---|---|
| baseline | nothing (plain PyTorch) | 44.84 | |
| BASIC | dmx-compressor's built-in `config_rules.BASIC` | 45.16 | +0.7% |
| BFP16_64 | Linear inputs and weights, 8-bit mantissa, block 64 | 44.99 | +0.3% |
| BFP14_64 | Linear inputs and weights, 6-bit mantissa, block 64 | 46.87 | +4.5% |
| BFP12_128 | Linear inputs and weights, 4-bit mantissa, block 128 | 344.70 | +669% |
| BFP12_64 | Linear inputs and weights, 4-bit mantissa, block 64 | 101.43 | +126% |
| BFP12_32 | Linear inputs and weights, 4-bit mantissa, block 32 | 71.79 | +60% |
| BFP12_16 | Linear inputs and weights, 4-bit mantissa, block 16 | 60.71 | +35% |
| BFP12_64 weights only | Linear weights only, 4-bit mantissa, block 64 | 57.09 | +27% |

What I took from it:
- BASIC mode (BFP16_64 matmuls plus FP16 elsewhere) costs well under 1% perplexity on this model.
- At 8 and 6 mantissa bits the block size barely matters. At 4 bits it dominates: shrinking the block from 128 to 16 takes perplexity from 345 down to 61, because one large value in a block forces a coarse shared exponent on everything else in it.
- At 4 bits, casting activations hurts much more than casting weights: weights-only BFP12_64 is 57.1, while weights plus inputs is 101.4.

I left out the plain fixed-point formats (`INT8`, `INT4` = `XP[8,0]`, `XP[4,0]`). They have no scale, so applying them directly to weights rounds almost everything to zero; they need a scale from an observer or a quantization recipe, which is a different experiment.

## Run it

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python bench.py --windows 16 --out results.json   # about 6 minutes on an Apple Silicon laptop CPU
python -m pytest -q tests
```

dmx-compressor compiles a small C++ extension on first import, so `ninja` and a C++ compiler need to be available. The model and dataset download from the Hugging Face Hub (no account needed).

The tests use a tiny randomly initialized OPT so they run in a few seconds: wrapping a model with no rules leaves its outputs unchanged, and a 4-bit-mantissa cast produces more error than an 8-bit one.

Tested with Python 3.14, torch 2.15.0, transformers from PyPI, dmx-compressor 0.1.11.
