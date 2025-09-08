import torch
from utils.profiler import Profiler
from models import resnet18, AlexNet

torch.random.manual_seed(0)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Build model and wrap with Profiler (module-level first, ops disabled)
backbone = resnet18().eval().to(device)
prof = Profiler(backbone, granularity="module", auto_profile=False, capture_ops=False)

# Prepare a dry-run input
x = torch.randn(1, 3, 224, 224, device=device)

# First pass: module-level graph and export
prof.profile_once(x)
prof.export("resnet18_profile", module_include_edges=True, include_stats=True, overwrite_module=True)

# Optional: second pass operator-level detail graph and export
prof.profile_detail(x)
prof.export("resnet18_profile", detail_include_edges=True, include_stats=True, overwrite_module=False)

# Print stats
print("==== Module Graph Stats ====")
prof.graph.stats()
if prof.detail_graph:
    print("==== Detail Graph Stats ====")
    prof.detail_graph.stats()

# Do a normal inference to ensure hooks are cleaned up
_ = prof(x)