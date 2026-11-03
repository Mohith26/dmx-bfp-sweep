import torch
from transformers import OPTConfig, OPTForCausalLM
from dmx.compressor import format as F
from dmx.compressor.modeling import DmxModel

import bench


def tiny_opt():
    torch.manual_seed(0)
    cfg = OPTConfig(vocab_size=128, hidden_size=64, ffn_dim=128, num_hidden_layers=2,
                    num_attention_heads=4, max_position_embeddings=64, word_embed_proj_dim=64)
    return OPTForCausalLM(cfg).eval()


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_wrapping_without_rules_keeps_outputs():
    m = tiny_opt()
    ids = torch.randint(0, 128, (1, 32))
    ref = logits(m, ids)
    dm = DmxModel.from_torch(m)
    assert torch.allclose(logits(dm, ids), ref, atol=1e-4)


def test_fewer_mantissa_bits_means_more_error():
    m = tiny_opt()
    ids = torch.randint(0, 128, (1, 32))
    ref = logits(m, ids)
    errs = {}
    for name in ["BFP16_64", "BFP12_64"]:
        dm = DmxModel.from_torch(tiny_opt())
        dm = dm.transform(dm.dmx_config, *bench.linear_rule(getattr(F, name)))
        errs[name] = (logits(dm, ids) - ref).abs().mean().item()
    assert 0 < errs["BFP16_64"] < errs["BFP12_64"]


def test_every_config_name_is_buildable():
    for name, rules in bench.CONFIGS.items():
        assert rules is None or len(rules) > 0, name
