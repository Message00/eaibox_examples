import torch
import transformers
import time, os, gc
from utils.profiler import Profiler
from graph import TensorWrapper
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])
model = transformers.LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# capture_ops=False records module-level only; set capture_ops=True to record operator-level
model.model = Profiler(model.model, granularity="module", token_kwarg="input_ids", auto_profile=False, capture_ops=False)
# First do a dry-run to build the module-level graph
dry_inputs = tokenizer("Hello", return_tensors="pt")
model.model.profile_once(**dry_inputs)
# Export module-level graph
model.model.export("llama_profile", module_include_edges=True, detail_include_edges=True, include_stats=True, overwrite_module=True)
# Generate operator-level detail graph (optional)
model.model.profile_detail(**dry_inputs)
model.model.export("llama_profile", module_include_edges=True, detail_include_edges=True, include_stats=True, overwrite_module=False)  # Second pass writes only detail
pipeline = transformers.pipeline(
    "text-generation", model=model, tokenizer=tokenizer, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

print(pipeline("Hey how are you doing today?"))
print("==== Module Graph Stats ====")
model.model.graph.stats()
if model.model.detail_graph:
    print("==== Detail Graph Stats ====")
    model.model.detail_graph.stats()