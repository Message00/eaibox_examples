import torch
from utils.profiler import Profiler
from model import ModelArgs, Transformer
import gc

torch.random.manual_seed(0)

# Build LLaMA model
model_args = ModelArgs(n_layers=8, vocab_size=32000, dim=8192, n_heads=64, n_kv_heads=8, multiple_of=4096, ffn_dim_multiplier=1.3)
net = Transformer(model_args).eval()
prof = Profiler(net, granularity="module", auto_profile=False, capture_ops=False, token_kwarg=None)

# Prepare a dry-run input (match your Transformer.forward signature)
x = torch.randint(1, 32000, (1, 1))

# Module-level graph and export
prof.profile_once(x, 1, 0)
prof.export("llama_profile", module_include_edges=True, include_stats=True, overwrite_module=True)

# Operator-level graph and export (optional)
#prof.profile_detail(x, 1, 0)
#prof.export("llama_profile", detail_include_edges=True, include_stats=True, overwrite_module=False)

# Print stats
print("==== Module Graph Stats ====")
prof.graph.stats()
if prof.detail_graph:
    print("==== Detail Graph Stats ====")
    prof.detail_graph.stats()

# Subsequent inference example
_ = prof(x, 1, 0)
del net
gc.collect()
