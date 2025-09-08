import torch
import torch.nn as nn
import collections
import time
from graph import Graph, Node, TensorWrapper, ModuleWrapper
try:
    from transformers.utils import ModelOutput
except Exception:
    class ModelOutput(dict):
        pass

def modify_forward(module):
    original_forward = module.forward
    # Save the original forward to restore later
    if not hasattr(module, "_orig_forward"):
        module._orig_forward = original_forward
    def modified_forward(x, *args, **kwargs):
        # Infer graph from input TensorWrapper first; fallback to module._prof_graph (injected during instrumentation)
        graph = x.graph if isinstance(x, TensorWrapper) else getattr(module, "_prof_graph", None)
        # Behavior per mode:
        # - Module-level (profile_once, capture_ops=False): unwrap inputs to avoid interference.
        # - Operator-level (profile_detail, capture_ops=True): let wrappers flow through modules
        #   to capture Python-level tensor ops like residual adds (submodules will unwrap inside their own modified_forward).
        def maybe_unwrap(v):
            if isinstance(v, TensorWrapper):
                if graph is not None and getattr(graph, 'capture_ops', False):
                    return v  # allow wrapper passthrough
                return v.tensor  # unwrap at module-level
            return v
        real_x = maybe_unwrap(x)
        real_args = tuple(maybe_unwrap(a) for a in args)
        real_kwargs = {k: maybe_unwrap(v) for k, v in kwargs.items()}
        out = original_forward(real_x, *real_args, **real_kwargs)
        # After graph is frozen (first pass complete), stop wrapping and return real output
        if graph is None or getattr(graph, 'is_freeze', False):
            return out
        # Standard tensor: wrap directly
        if isinstance(out, torch.Tensor):
            tw = TensorWrapper(out, graph)
            # In operator-level graph, inherit input last_node to keep chain continuity
            if isinstance(x, TensorWrapper) and getattr(graph, 'capture_ops', False):
                tw.last_node = x.last_node
            return tw
        # Container output: in tuple/list, pick the first primary activation (first torch.Tensor),
        # wrap only that element to keep the main path chain; keep others as-is
        if isinstance(out, (list, tuple)):
            wrapped_flag = False
            new_elems = []
            for elem in out:
                if (not wrapped_flag) and isinstance(elem, torch.Tensor):
                    tw = TensorWrapper(elem, graph)
                    new_elems.append(tw)
                    wrapped_flag = True
                else:
                    new_elems.append(elem)
            if isinstance(out, tuple):
                return tuple(new_elems)
            return new_elems
        # ModelOutput or dict: similarly wrap only the first field whose value is a tensor
        try:
            from transformers.utils import ModelOutput
            model_output_cls = ModelOutput
        except Exception:
            model_output_cls = tuple()  # placeholder
        if isinstance(out, model_output_cls):
            data_items = list(out.items())
            wrapped_flag = False
            new_dict = {}
            for k, v in data_items:
                if (not wrapped_flag) and isinstance(v, torch.Tensor):
                    tw = TensorWrapper(v, graph)
                    if isinstance(x, TensorWrapper) and getattr(graph, 'capture_ops', False):
                        tw.last_node = x.last_node
                    new_dict[k] = tw
                    wrapped_flag = True
                else:
                    new_dict[k] = v
            try:
                return out.__class__(**new_dict)
            except Exception:
                return new_dict
        if isinstance(out, dict):
            wrapped_flag = False
            for k, v in list(out.items()):
                if (not wrapped_flag) and isinstance(v, torch.Tensor):
                    tw = TensorWrapper(v, graph)
                    if isinstance(x, TensorWrapper) and getattr(graph, 'capture_ops', False):
                        tw.last_node = x.last_node
                    out[k] = tw
                    wrapped_flag = True
            return out
        # Other types: return as-is
        return out
    module.forward = modified_forward

class Profiler(nn.Module):
    def __init__(self, model: nn.Module, granularity: str = "layer", token_kwarg: str = None, auto_profile: bool = True, capture_ops: bool = False):
        super().__init__()
        # Primary graph (module-level)
        self.graph = Graph()
        self.graph.capture_ops = capture_ops  # whether to capture ops in the first pass
        self.tensorwrapper_buffer = None
        self.granularity = granularity
        self.token_kwarg = token_kwarg

        if granularity not in ["layer", "module"]:
            raise ValueError("Granularity must be either 'layer' or 'module'")

        # Model & runtime state
        self.model = model
        self.nn_module_ids = []
        self.queue = collections.deque([])
        self.i = 0
        self.init_time = -1
        self.ttl_infer_time = 0.0
        self.initial_forward_flag = True
        self.forward_counter = 0

        # Flags
        self.instrumented = False
        self.profiled = False
        self.detail_graph = None  # operator-level detail graph (second pass)
        self.backward_time = []  # avoid uninitialized reference

        # Hook handles
        self._wrapped_modules = []
        self._pre_hook_handles = []
        self._post_hook_handles = []

        # Optionally instrument immediately
        if auto_profile:
            self._instrument_model()

    def layer_forward_time_start(self, layer, input):
        # Skip in operator-level graph (no module nodes recorded there)
        if getattr(self.graph, 'capture_ops', False):
            return
        use_cuda = torch.cuda.is_available()
        now = lambda: time.time() * 1000
        if self.initial_forward_flag:
            params = 0
            for param_name, param in layer.named_parameters():
                params += torch.prod(torch.LongTensor(list(param.size())))

            layer_args = vars(layer).copy()
            keys_to_remove = [k for k in layer_args.keys() if k.startswith("_")]
            for k in keys_to_remove:
                del layer_args[k]
            if "forward" in layer_args:
                del layer_args["forward"]

            node = Node(self.graph, self.graph.get_new_id(), layer.__class__.__name__, layer.__class__, layer_args)
            node.param_size = int(params)
            self.nn_module_ids.append(node.id)
            self.graph.add_nn_node(node)

            # Connect from previous nodes: scan all inputs (including nested structures)
            # and collect TensorWrappers that have last_node
            def collect_wrappers(obj, acc):
                if isinstance(obj, TensorWrapper) and obj.last_node is not None:
                    acc.append(obj)
                elif isinstance(obj, (list, tuple)):
                    for e in obj:
                        collect_wrappers(e, acc)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        collect_wrappers(v, acc)
            wrappers = []
            for inp in input:
                collect_wrappers(inp, wrappers)
            added_sources = set()
            for w in wrappers:
                if w.last_node.id not in added_sources:
                    self.graph.add_edge(w.last_node, node)
                    added_sources.add(w.last_node.id)
            if use_cuda:
                torch.cuda.synchronize()
            node.forward_time -= now()
        else:
            node = self.graph.get_node(self.nn_module_ids[self.forward_counter])
            self.forward_counter += 1
            if use_cuda:
                torch.cuda.synchronize()
            node.forward_time -= now()

    def layer_forward_time_end(self, layer: nn.Module, input, output):
        # Skip in operator-level graph (no module nodes recorded there)
        if getattr(self.graph, 'capture_ops', False):
            return
        use_cuda = torch.cuda.is_available()
        now = lambda: time.time() * 1000
        if self.initial_forward_flag:
            node = self.graph.get_node(self.nn_module_ids[-1])  # the node just created
            if use_cuda:
                torch.cuda.synchronize()
            if node is not None:
                node.forward_time += now()
                node.forward_epoch += 1
            # Set last_node for the primary TensorWrapper in the output
            def assign_last_node(o):
                if isinstance(o, TensorWrapper):
                    o.last_node = node
                    return True
                if isinstance(o, (list, tuple)):
                    for e in o:
                        if assign_last_node(e):
                            return True
                if isinstance(o, dict):
                    for v in o.values():
                        if assign_last_node(v):
                            return True
                return False
            assign_last_node(output)
        else:
            node = self.graph.get_node(self.nn_module_ids[self.forward_counter - 1])
            if use_cuda:
                torch.cuda.synchronize()
            if node is not None:
                node.forward_time += now()
                node.forward_epoch += 1

    def _cleanup_instrumentation(self):
        """Restore original forwards for all modules and remove hooks to avoid later interference with TensorWrapper."""
        # Restore forward
        for module in self.model.modules():
            if hasattr(module, "_orig_forward"):
                module.forward = module._orig_forward
            if hasattr(module, "_prof_graph"):
                try:
                    delattr(module, "_prof_graph")
                except Exception:
                    pass
        # Remove forward hooks: we stored handles during registration
        for h in getattr(self, "_pre_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        for h in getattr(self, "_post_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        self._pre_hook_handles = []
        self._post_hook_handles = []
        self.instrumented = False

    def _instrument_model(self):
        if self.instrumented:
            return
        layer_idx = 0
        print("Profiling layers...")
        total_params = 0
        trainable_params = 0
        print(f"{'Layer ID':>8}{'Layer Name':>30}{'Params':>12}{'Trainable':>12}")
        self.queue.clear()
        for name, module in self.model.named_children():
            self.queue.append((name, module))
            while len(self.queue) > 0:
                node = list(self.queue.popleft())
                flag = True
                new_queue = []
                if self.granularity == "module":
                    if not node[0].split('.')[-1].isdigit():
                        for n, mod in node[1].named_children():
                            flag = False
                            new_queue.append((node[0] + "." + n, mod))
                elif self.granularity == "layer":
                    for n, mod in node[1].named_children():
                        flag = False
                        new_queue.append((node[0] + "." + n, mod))
                if flag:
                    if isinstance(node[1], nn.ReLU):
                        node[1].__init__(inplace=False)
                    layer_idx += 1
                    params = 0
                    required_grad = False
                    for param_name, param in node[1].named_parameters():
                        if 'weight' in param_name:
                            params += torch.prod(torch.LongTensor(list(param.size())))
                            if param.requires_grad:
                                required_grad = True
                        elif 'bias' in param_name:
                            params += torch.prod(torch.LongTensor(list(param.size())))
                    if required_grad:
                        trainable_params += params
                    total_params += params
                    print(f"{layer_idx:>8}{node[0]:>30}{params:>12}{required_grad:>12b}")
                    modify_forward(node[1])
                    # Inject current graph context into the module for fallback when there is no Wrapper input inside
                    try:
                        setattr(node[1], "_prof_graph", self.graph)
                    except Exception:
                        pass
                    pre_h = node[1].register_forward_pre_hook(self.layer_forward_time_start)
                    post_h = node[1].register_forward_hook(self.layer_forward_time_end)
                    self._pre_hook_handles.append(pre_h)
                    self._post_hook_handles.append(post_h)
                else:
                    new_queue.reverse()
                    self.queue.extendleft(new_queue)
        print(f"Total params: {total_params} - Trainable params: {trainable_params}")
        self.instrumented = True
        # Record wrapped modules
        self._wrapped_modules = []
        for m in self.model.modules():
            if hasattr(m, "_orig_forward") and m.forward != getattr(m, "_orig_forward"):
                self._wrapped_modules.append(m)

    def profile_once(self, *args, **kwargs):
        """Run a single dry-forward to build the module-level graph, then remove instrumentation."""
        if self.profiled:
            return
        # Initialize state
        self.initial_forward_flag = True
        self.forward_counter = 0
        # Reset primary graph (module-level) and disable op capture
        self.graph = Graph()
        self.graph.capture_ops = False
        self.nn_module_ids = []
        # Instrument
        self._instrument_model()
        # Run one forward (internal forward will freeze + cleanup)
        _ = self.forward(*args, **kwargs)
        self.profiled = True
        # instrumentation was cleaned up inside forward

    def profile_detail(self, *args, **kwargs):
        """Generate an operator-level detail graph (second independent forward, preserving the module-level graph)."""
        if not self.profiled:
            raise RuntimeError("Call profile_once() first to build the module-level graph before calling profile_detail().")
        if self.detail_graph is not None:
            return
        # Create a new operator-level graph
        self.detail_graph = Graph()
        self.detail_graph.capture_ops = True
        original_graph = self.graph
        self.graph = self.detail_graph
        # Re-instrument
        self.initial_forward_flag = True
        self.forward_counter = 0
        self.nn_module_ids = []
        self._instrument_model()
        try:
            self.forward(*args, **kwargs)
        finally:
            self.graph = original_graph

    def get_graphs(self):
        return {"module_graph": self.graph, "detail_graph": self.detail_graph}

    def export(self, base_path: str = "profile", 
               module_include_edges: bool = True, detail_include_edges: bool = True,
               include_stats: bool = True, overwrite_module: bool = False):
        module_file = f"{base_path}_module.txt"
        detail_file = f"{base_path}_detail.txt"
        if self.graph is not None:
            import os
            if overwrite_module or (not os.path.exists(module_file)):
                self.graph.export(module_file, include_edges=module_include_edges, include_stats=include_stats)
        if self.detail_graph is not None:
            self.detail_graph.export(detail_file, include_edges=detail_include_edges, include_stats=include_stats)
    def layer_backward_time_start(self, layer, input):
        torch.cuda.synchronize()
        self.backward_time.append(time.time())

    def layer_backward_time_end(self, layer, input, output):
        torch.cuda.synchronize()
        self.backward_time.append(time.time())
        print(f"Layer: {layer} - backward time: {(self.backward_time[-1] - self.backward_time[-2]) * 1000} ms")

    def model_forward_pre_hook(self, module, input):
        torch.cuda.synchronize()
        self.init_time = time.time()

    def model_forward_hook(self, module, input, output): 
        for layer in self.layer_dict.keys():
            fw_time = self.layer_dict[layer]
            if fw_time < 1000:
                print(f"Layer {layer:<30} - forward time: {fw_time} ms")
                pass
            else:
                print(f"Layer {layer:<30} - forward time: {fw_time / 1000} s")
                pass

    def forward(self, *args, **kwargs):
        def unwrap_output(o):
            if isinstance(o, TensorWrapper):
                return o.tensor
            if isinstance(o, (list, tuple)):
                return type(o)(unwrap_output(e) for e in o)
            if isinstance(o, ModelOutput):
                new_data = {k: unwrap_output(v) for k,v in o.items()}
                try:
                    return o.__class__(**new_data)
                except Exception:
                    # Fallback to dict if reconstruction fails
                    return new_data
            if isinstance(o, dict):
                return {k: unwrap_output(v) for k,v in o.items()}
            return o

        output = None
        self.forward_counter = 0
        tokens = kwargs.get(self.token_kwarg, None) if self.token_kwarg else (args[0] if len(args) > 0 else None)
        # If not instrumented yet (dry mode before profile_once), directly call the model
        if not self.instrumented:
            return self.model(*args, **kwargs)
        if self.initial_forward_flag:
            try:
                if isinstance(tokens, torch.Tensor):
                    tw_tokens = TensorWrapper(tokens, self.graph)
                    self.graph.add_opr_node("Input", None)
                    tw_tokens.last_node = self.graph.nodes[-1]
                    if self.token_kwarg:
                        kwargs[self.token_kwarg] = tw_tokens
                    else:
                        args = (tw_tokens,) + args[1:]
                for i, arg in enumerate(args):
                    if isinstance(arg, TensorWrapper):
                        self.graph.add_opr_node("Input", None)
                        arg.last_node = self.graph.nodes[-1]
                output = self.model(*args, **kwargs)
                if isinstance(output, TensorWrapper):
                    output.last_node = self.graph.get_node(self.graph.counter)
                    self.graph.add_opr_node("Output", None)
                    self.graph.add_edge(output.last_node, self.graph.nodes[-1])
            except Exception as e:
                if str(e).startswith("flatten()"):
                    print("Error: Please use torch.Tensor.view() instead of torch.flatten()")
                else:
                    import traceback
                    traceback.print_exc()
            self.initial_forward_flag = False
            self.graph.freeze()
            # Cleanup instrumentation, restore original forwards, prevent wrappers from affecting subsequent generation
            self._cleanup_instrumentation()
            return unwrap_output(output)
        else:
            output = self.model(*args, **kwargs)
            return unwrap_output(output)
    
    def clear_model(self):
        del self.model
    
if __name__ == '__main__':
    from models import Transformer, ModelArgs
    model_args = ModelArgs(n_layers=8, vocab_size=32000)
    model = Transformer(model_args)
    model = Profiler(model, granularity="module")
    input = torch.randint(0, 32000, (4, 1))
    input = TensorWrapper(input, model.graph)
    initial_node = Node(model.graph, 0, "Input", None, None)
    model.graph.add_nn_node(initial_node)
    input.last_node = initial_node
    output = model(input, 0)
    model.graph.summary()
