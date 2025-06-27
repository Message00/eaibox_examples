from edgespeed.llama.inference import LLaMA
from edgespeed.pipe import PipelineModule
import edgespeed

ckpt_dir = '/home/eaibox/Llama-2-7b/'
tokenizer_path = '/home/eaibox/Llama-2-7b/tokenizer.model'

llama = LLaMA.build(
    checkpoints_dir = ckpt_dir,
    tokenizer_path= tokenizer_path,
    load_model=True,
    max_seq_len=64,
    max_batch_size=2,
    device='cpu'
)

distributor = edgespeed.init_distributed(['master', 'worker0', 'worker1', 'worker2'], '192.168.1.101', '5678')
layers = llama.model.get_layers()
model2 = PipelineModule(
    distributor=distributor,
    nodes=['master', 'worker0', 'worker1', 'worker2'],
    layers=layers,
    partition_method='parameters',
    forward_with_micro_batch_offset=True
    )
llama.model = model2

results = llama.text_completion(["What can I say?"], 'cpu', temperature=0, max_gen_len=32)
print(results)
    
distributor.shutdown_rpc()
