# EdgeShard Profiler

A lightweight, dry-run graph profiler for PyTorch models (and optional Hugging Face LLaMA). It builds a module-level graph first, then an operator-level detail graph on a second pass, exporting nodes, edges, and timing to text files.

## Features

- Module-level graph with per-layer forward time and parameter counts
- Optional operator-level detail graph (op-only, module nodes excluded)
- Complete edges across modules and ops (residual adds, Python-level tensor ops, etc.)
- Export to human-readable text with full edge lists and summary stats
- Works with plain PyTorch nn.Module and HF LLaMA (token_kwarg supported)

## Requirements

- Python 3.8+
- PyTorch
- Optional: transformers (only for HF examples)

## Quick start (any PyTorch model)

Example using ResNet18 (see `test_tensor.py`):

```python
import torch
from utils.profiler import Profiler
from models import resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet18().eval().to(device)
prof = Profiler(model, granularity="module", auto_profile=False, capture_ops=False)

x = torch.randn(1, 3, 224, 224, device=device)

# Pass 1: module-level graph
prof.profile_once(x)
prof.export("resnet18_profile", module_include_edges=True, include_stats=True, overwrite_module=True)

# Optional Pass 2: operator-level detail graph
prof.profile_detail(x)
prof.export("resnet18_profile", detail_include_edges=True, include_stats=True, overwrite_module=False)

# Print stats in console
print("==== Module Graph Stats ====")
prof.graph.stats()
if prof.detail_graph:
    print("==== Detail Graph Stats ====")
    prof.detail_graph.stats()

# Normal inference works as usual (hooks are cleaned up automatically)
_ = prof(x)
```

Notes:
- Pass 1 freezes and cleans up instrumentation automatically after the forward finishes.
- Operator-level detail graph excludes module nodes; “noise” ops that do not return tensors (e.g., `dim`, `item`) are filtered out.

## Hugging Face LLaMA (optional)

You can wrap `model.model` of a HF `LlamaForCausalLM` with Profiler and point `token_kwarg` to the input IDs key. See `test_hf_llama.py` for a complete example. A minimal sketch:

```python
import torch, transformers
from utils.profiler import Profiler

model = transformers.LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Wrap the inner model and indicate which kwarg carries tokens
model.model = Profiler(model.model, granularity="module", token_kwarg="input_ids", auto_profile=False, capture_ops=False)

dry_inputs = tokenizer("Hello", return_tensors="pt")

# Module-level graph
model.model.profile_once(**dry_inputs)
model.model.export("llama_profile", module_include_edges=True, include_stats=True, overwrite_module=True)

# Optional operator-level detail graph
model.model.profile_detail(**dry_inputs)
model.model.export("llama_profile", detail_include_edges=True, include_stats=True, overwrite_module=False)
```

Caveats:
- HF models are large; ensure sufficient disk/GPU memory.
- Do not hardcode access tokens in code or files.

## API overview

- Profiler(model, granularity="module"|"layer", token_kwarg: str|None = None, auto_profile=True, capture_ops=False)
  - granularity:
    - "module": coarser grouping by high-level submodules
    - "layer": expand to leaf layers
  - token_kwarg: name of the kwarg that carries input tokens (for HF-style forward signatures)
  - capture_ops: whether to attempt operator capture in the first pass (default False; detail pass always captures ops)
- profile_once(*args, **kwargs): build the module-level graph with a single forward
- profile_detail(*args, **kwargs): run a second forward to build an operator-level detail graph
- export(base_path, module_include_edges=True, detail_include_edges=True, include_stats=True, overwrite_module=False)
  - Writes `base_path_module.txt` and/or `base_path_detail.txt`
- get_graphs(): returns a dict `{ "module_graph": Graph, "detail_graph": Graph|None }`
- Forward usage: you can call the Profiler instance like the original model after profiling; wrappers/hooks are cleaned up.

## Outputs

Two text files by default:
- `<base>_module.txt`: module nodes with ids, names, parameter sizes, forward times, and (if enabled) edges
- `<base>_detail.txt`: operator-only nodes with per-op forward times and complete edges across ops

Both files include a compact stats section (total nodes/edges and forward-time aggregates) when `include_stats=True`.

## Tips & limitations

- Timing uses wall-clock with `torch.cuda.synchronize()` when CUDA is available; numbers are indicative, not micro-benchmarks.
- Operator capture focuses on Python-level tensor ops; low-level ATen kernels (e.g., conv2d/addmm) are represented by their high-level calls.
- Non-tensor-returning calls (e.g., `dim`, `item`) are not recorded as nodes in the detail graph to reduce noise.
- If you see an error about `flatten()`, prefer `tensor.view(...)` within models.

## Examples in repo

- `test_tensor.py`: ResNet18 example
- `test_llama.py`: Local Transformer example
- `test_llamab.py`: Multi-epoch timing sketch for a local LLaMA-like model
- `test_hf_llama.py`: Hugging Face LLaMA integration (downloads a large model)

---

If you run into issues or need deeper capture (e.g., ATen-level ops), open an issue or extend the hooks as needed.
