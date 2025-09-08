import torch
from utils.profiler import Profiler
from models import ModelArgs, Transformer
import time, gc
from tqdm import tqdm

torch.random.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
device = "cuda" if torch.cuda.is_available() else "cpu"

for layer_num in [13]:
    layer_result = []
    print("Initializing model...")
    model_args = ModelArgs(n_layers=layer_num, vocab_size=32000, dim=4096, n_heads=32, multiple_of=256)
    net = Transformer(model_args).eval().to(device)
    prof = Profiler(net, granularity="module", auto_profile=False, capture_ops=False)
    print("Model initialized.")

    # Do a module-level dry run first for later epoch stats
    dry = torch.randint(0, 32000, (1, 1), device=device)
    prof.profile_once(dry, 0)

    loader = tqdm(range(80))
    for epoch in loader:
        loader.set_description(f"Layer: {layer_num} - Epoch: {epoch}")
        x = torch.randint(0, 32000, (1, 1), device=device)
        start_time = time.time()
        _ = prof(x, epoch)
        once_time = prof.graph.get_forward_time_list()
        total = prof.graph.get_total_forward_time()
        layer_result.append(total)

    # Export module-level and (optional) operator-level
    prof.export("llamab_profile", module_include_edges=True, include_stats=True, overwrite_module=True)
    # If details are needed:
    prof.profile_detail(dry, 0)
    prof.export("llamab_profile", detail_include_edges=True, include_stats=True, overwrite_module=False)

    prof.graph.stats()
    del prof, net, model_args
    gc.collect()

