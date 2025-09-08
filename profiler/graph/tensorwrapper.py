import torch
import time
class TensorWrapper:
    _TORCH_HANDLED = {torch.cat, torch.stack}

    def __init__(self, tensor: torch.Tensor, graph):
        self.tensor = tensor
        self.last_node = None
        self.graph = graph

    @property
    def shape(self):
        return self.tensor.shape
    
    @property
    def device(self):
        return self.tensor.device
    
    def reshape(self, *wrapped_sizes):
        if self.graph is None or self.graph.is_freeze or not getattr(self.graph, 'capture_ops', True):
            sizes = [ws.tensor if isinstance(ws, TensorWrapper) else ws for ws in wrapped_sizes]
            result = self.tensor.reshape(*sizes)
            # Module-level passthrough: wrap and inherit last_node to keep the chain
            if self.graph is not None and self.last_node is not None:
                w = TensorWrapper(result, self.graph)
                w.last_node = self.last_node
                return w
            return result
        sizes = []
        for wrapped_size in wrapped_sizes:
            if isinstance(wrapped_size, TensorWrapper):
                sizes.append(wrapped_size.tensor)
            else:
                sizes.append(wrapped_size)
        node = self._op_start("Reshape", wrapped_sizes)
        result = self.tensor.reshape(*sizes)
        wrapped_result = TensorWrapper(result, self.graph)
        if node is None:
            return wrapped_result
        wrapped_result.last_node = node
        if self.last_node is not None:
            self.graph.add_edge(self.last_node, node)
        self._op_end(node)
        return wrapped_result
    
    def float(self):
        return TensorWrapper(self.tensor.float(), self.graph)

    def view(self, *wrapped_sizes):
        if self.graph is None or self.graph.is_freeze or not getattr(self.graph, 'capture_ops', True):
            sizes = [ws.tensor if isinstance(ws, TensorWrapper) else ws for ws in wrapped_sizes]
            result = self.tensor.view(*sizes)
            if self.graph is not None and self.last_node is not None:
                w = TensorWrapper(result, self.graph)
                w.last_node = self.last_node
                return w
            return result
        sizes = []
        for wrapped_size in wrapped_sizes:
            if isinstance(wrapped_size, TensorWrapper):
                sizes.append(wrapped_size.tensor)
            else:
                sizes.append(wrapped_size)
        node = self._op_start("View", wrapped_sizes)
        result = self.tensor.view(*sizes)
        wrapped_result = TensorWrapper(result, self.graph)
        if node is None:
            return wrapped_result
        wrapped_result.last_node = node
        if self.last_node is not None:
            self.graph.add_edge(self.last_node, node)
        self._op_end(node)
        return wrapped_result
        
    def __gt__(self, other):
        return self.tensor.__gt__(other)

    def __lt__(self, other):
        return self.tensor.__lt__(other)

    def __add__(self, other):
        if self.graph is None or self.graph.is_freeze or not getattr(self.graph, 'capture_ops', True):
            return self.tensor + (other.tensor if isinstance(other, TensorWrapper) else other)
        if isinstance(other, TensorWrapper):
            args = (other.last_node.id if other.last_node else None, self.last_node.id if self.last_node else None)
            node = self._op_start("Add", args)
            result_tensor = self.tensor + other.tensor
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if other.last_node:
                    self.graph.add_edge(other.last_node, node)
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result
        else:
            args = (other, self.last_node.id if self.last_node else None)
            node = self._op_start("Add", args)
            result_tensor = self.tensor + other
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result

    def __iadd__(self, other):
        if isinstance(other, TensorWrapper):
            args = (other.last_node.id if other.last_node else None, self.last_node.id if self.last_node else None)
            node = self._op_start("Add(inplace)", args)
            result_tensor = self.tensor + other.tensor
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if other.last_node:
                    self.graph.add_edge(other.last_node, node)
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result
        else:
            args = (other, self.last_node.id if self.last_node else None)
            node = self._op_start("Add(inplace)", args)
            result_tensor = self.tensor + other
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result
    
    def __sub__(self, other):
        if self.graph is None or self.graph.is_freeze or not getattr(self.graph, 'capture_ops', True):
            return self.tensor - (other.tensor if isinstance(other, TensorWrapper) else other)
        if isinstance(other, TensorWrapper):
            args = (other.last_node.id if other.last_node else None, self.last_node.id if self.last_node else None)
            node = self._op_start("Sub", args)
            result_tensor = self.tensor - other.tensor
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if other.last_node:
                    self.graph.add_edge(other.last_node, node)
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result
        else:
            args = (other, self.last_node.id if self.last_node else None)
            node = self._op_start("Sub", args)
            result_tensor = self.tensor - other
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result
    
    def __isub__(self, other):
        if isinstance(other, TensorWrapper):
            self.tensor -= other.tensor
            self.graph.add_opr_node("Sub(inplace)", (other.last_node.id, self.last_node.id))
        else:
            self.tensor -= other
            self.graph.add_opr_node("Sub(inplace)", (other, self.last_node.id))
        return self
    
    def __neg__(self):
        result_tensor = -self.tensor
        if self.graph is None or self.graph.is_freeze or self.last_node is None:
            return result_tensor
        self.graph.add_opr_node("Neg", (self.last_node.id,))
        wrapped_result = TensorWrapper(result_tensor, self.graph)
        wrapped_result.last_node = self.graph.nodes[-1]
        self.graph.add_edge(self.last_node, wrapped_result.last_node)
        return wrapped_result

    def __mul__(self, other):
        if self.graph is None or self.graph.is_freeze or not getattr(self.graph, 'capture_ops', True):
            return self.tensor * (other.tensor if isinstance(other, TensorWrapper) else other)
        if isinstance(other, TensorWrapper):
            args = (other.last_node.id if other.last_node else None, self.last_node.id if self.last_node else None)
            node = self._op_start("Mul", args)
            result_tensor = self.tensor * other.tensor
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if other.last_node:
                    self.graph.add_edge(other.last_node, node)
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result
        else:
            args = (other, self.last_node.id if self.last_node else None)
            node = self._op_start("Mul", args)
            result_tensor = self.tensor * other
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result

    def __truediv__(self, other):
        if self.graph is None or self.graph.is_freeze or not getattr(self.graph, 'capture_ops', True):
            return self.tensor / (other.tensor if isinstance(other, TensorWrapper) else other)
        if isinstance(other, TensorWrapper):
            args = (other.last_node.id if other.last_node else None, self.last_node.id if self.last_node else None)
            node = self._op_start("Div", args)
            result_tensor = self.tensor / other.tensor
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if other.last_node:
                    self.graph.add_edge(other.last_node, node)
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result
        else:
            args = (other, self.last_node.id if self.last_node else None)
            node = self._op_start("Div", args)
            result_tensor = self.tensor / other
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result

    def __rtruediv__(self, other):
        if self.graph is None or self.graph.is_freeze or not getattr(self.graph, 'capture_ops', True):
            return (other.tensor if isinstance(other, TensorWrapper) else other) / self.tensor
        if isinstance(other, TensorWrapper):
            args = (self.last_node.id if self.last_node else None, other.last_node.id if other.last_node else None)
            node = self._op_start("Div", args)
            result_tensor = other.tensor / self.tensor
            source_last = other.last_node
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None:
                wrapped_result.last_node = node
                if source_last is not None:
                    self.graph.add_edge(source_last, node)
                if self.last_node is not None:
                    self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result
        else:
            args = (self.last_node.id if self.last_node else None, other)
            node = self._op_start("Div", args)
            result_tensor = other / self.tensor
            wrapped_result = TensorWrapper(result_tensor, self.graph)
            if node is not None and self.last_node is not None:
                wrapped_result.last_node = node
                self.graph.add_edge(self.last_node, node)
                self._op_end(node)
            return wrapped_result

    def __getitem__(self, key):
        if self.graph is None or self.graph.is_freeze or not getattr(self.graph, 'capture_ops', True) or self.last_node is None:
            result = self.tensor[key]
            if isinstance(result, torch.Tensor) and self.graph is not None and self.last_node is not None and not getattr(self.graph, 'capture_ops', True):
                w = TensorWrapper(result, self.graph)
                w.last_node = self.last_node
                return w
            return result
        node = self._op_start("GetItem", key)
        result_tensor = self.tensor[key]
        wrapped_result = TensorWrapper(result_tensor, self.graph)
        if node is None:
            return wrapped_result
        wrapped_result.last_node = node
        self.graph.add_edge(self.last_node, node)
        self._op_end(node)
        return wrapped_result
    
    def __repr__(self):
        return self.tensor.__repr__()

    def transpose(self, *args):
        if self.graph is None or self.graph.is_freeze or self.last_node is None or not getattr(self.graph, 'capture_ops', True):
            result = self.tensor.transpose(*args)
            if isinstance(result, torch.Tensor) and self.graph is not None and self.last_node is not None and not getattr(self.graph, 'capture_ops', True):
                w = TensorWrapper(result, self.graph)
                w.last_node = self.last_node
                return w
            return result
        args_str = ", ".join([str(arg) for arg in args])
        node = self._op_start("Transpose", args_str)
        result_tensor = self.tensor.transpose(*args)
        wrapped_result = TensorWrapper(result_tensor, self.graph)
        if node is None:
            return wrapped_result
        wrapped_result.last_node = node
        self.graph.add_edge(self.last_node, node)
        self._op_end(node)
        return wrapped_result

    def unsqueeze(self, *args):
        if self.graph is None or self.graph.is_freeze or self.last_node is None or not getattr(self.graph, 'capture_ops', True):
            result = self.tensor.unsqueeze(*args)
            if isinstance(result, torch.Tensor) and self.graph is not None and self.last_node is not None and not getattr(self.graph, 'capture_ops', True):
                w = TensorWrapper(result, self.graph)
                w.last_node = self.last_node
                return w
            return result
        node = self._op_start("Unsqueeze", args)
        result_tensor = self.tensor.unsqueeze(*args)
        wrapped_result = TensorWrapper(result_tensor, self.graph)
        if node is None:
            return wrapped_result
        wrapped_result.last_node = node
        self.graph.add_edge(self.last_node, node)
        self._op_end(node)
        return wrapped_result

    def size(self, *args):
        return self.tensor.size(*args)

    def to_tensor(self):
        return self.tensor

    def __len__(self):
        return len(self.tensor)

    @property
    def dtype(self):
        return self.tensor.dtype

    def to(self, *args, **kwargs):
        node = None
        if self.graph is not None and not self.graph.is_freeze and self.last_node is not None and getattr(self.graph, 'capture_ops', True):
            node = self._op_start("To", (self.last_node.id,))
        result_tensor = self.tensor.to(*args, **kwargs)
        wrapped_result = TensorWrapper(result_tensor, self.graph) if self.graph is not None else result_tensor
        if node is not None:
            wrapped_result.last_node = node
            self.graph.add_edge(self.last_node, node)
            self._op_end(node)
        elif self.graph is not None and self.last_node is not None and not getattr(self.graph, 'capture_ops', True):
            # Module-level passthrough
            wrapped_result.last_node = self.last_node
        return wrapped_result

    def clone(self, *args, **kwargs):
        if self.graph is None or self.graph.is_freeze or self.last_node is None or not getattr(self.graph, 'capture_ops', True):
            result = self.tensor.clone(*args, **kwargs)
            if isinstance(result, torch.Tensor) and self.graph is not None and self.last_node is not None and not getattr(self.graph, 'capture_ops', True):
                w = TensorWrapper(result, self.graph)
                w.last_node = self.last_node
                return w
            return result
        node = self._op_start("Clone", (self.last_node.id,))
        result_tensor = self.tensor.clone(*args, **kwargs)
        wrapped_result = TensorWrapper(result_tensor, self.graph)
        if node is None:
            return wrapped_result
        wrapped_result.last_node = node
        self.graph.add_edge(self.last_node, node)
        self._op_end(node)
        return wrapped_result

    def detach(self):
        if self.graph is None or self.graph.is_freeze or self.last_node is None or not getattr(self.graph, 'capture_ops', True):
            result = self.tensor.detach()
            if isinstance(result, torch.Tensor) and self.graph is not None and self.last_node is not None and not getattr(self.graph, 'capture_ops', True):
                w = TensorWrapper(result, self.graph)
                w.last_node = self.last_node
                return w
            return result
        node = self._op_start("Detach", (self.last_node.id,))
        result_tensor = self.tensor.detach()
        wrapped_result = TensorWrapper(result_tensor, self.graph)
        if node is None:
            return wrapped_result
        wrapped_result.last_node = node
        self.graph.add_edge(self.last_node, node)
        self._op_end(node)
        return wrapped_result

    def __getattr__(self, name):
        # Delegate unknown attributes/methods to underlying tensor.
        # Avoid recursion for special attributes
        if name in {"tensor", "graph", "last_node", "_TORCH_HANDLED"}:
            return super().__getattribute__(name)
        attr = getattr(self.tensor, name)
        if callable(attr):
            def method(*args, **kwargs):
                # Create a node only if the return value is a Tensor; for non-tensors (dim, item, etc.),
                # pass through without adding to the graph
                use_node = (self.graph is not None and not self.graph.is_freeze and self.last_node is not None and getattr(self.graph, 'capture_ops', True))
                if use_node and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_ms = time.time() * 1000.0 if use_node else None
                result = attr(*args, **kwargs)
                if isinstance(result, torch.Tensor) and use_node:
                    # Create node after the real call and record timing
                    self.graph.add_opr_node(name, (self.last_node.id,))
                    node = self.graph.nodes[-1]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_ms = time.time() * 1000.0
                    node.forward_time += (end_ms - (start_ms or end_ms))
                    node.forward_epoch += 1
                    wrapped_result = TensorWrapper(result, self.graph)
                    wrapped_result.last_node = node
                    self.graph.add_edge(self.last_node, node)
                    return wrapped_result
                return result
            return method
        return attr
    
    # ===== the key part =====
    def __torch_function__(self, func, types, args=(), kwargs=None):
        """
        Intercept torch API calls like torch.cat([...]) when any argument is a TensorWrapper.
        """
        if kwargs is None:
            kwargs = {}

        # Only handle calls where our type participates, and only for selected funcs
        # if not any(issubclass(t, TensorWrapper) for t in types):
        #     return NotImplemented
        # if func not in self._TORCH_HANDLED:
        #     return NotImplemented

        # -- helpers ----------------------------------------------------------
        def unwrap(x):
            # Turn TensorWrapper(s) into bare torch.Tensor(s), recursively through containers
            if isinstance(x, TensorWrapper):
                return x.tensor
            if isinstance(x, (list, tuple)):
                return type(x)(unwrap(e) for e in x)
            if isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            return x

        def collect_wrappers(x, out):
            # Collect all TensorWrapper inputs (for graph edges)
            if isinstance(x, TensorWrapper):
                out.append(x)
            elif isinstance(x, (list, tuple)):
                for e in x:
                    collect_wrappers(e, out)
            elif isinstance(x, dict):
                for v in x.values():
                    collect_wrappers(v, out)

        # --------------------------------------------------------------------
        # 1) collect inputs for graph bookkeeping
        inputs = []
        collect_wrappers(args, inputs)
        collect_wrappers(kwargs, inputs)

        # 2) create graph node for this op (before running the real op)
        new_node = None
        capturing = (self.graph is not None and getattr(self.graph, 'capture_ops', True) and not self.graph.is_freeze)
        if capturing and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_ms = time.time() * 1000.0 if capturing else None

        # 3) call the real torch function on unwrapped args
        # NOTE: previously we passed the entire unwrapped args tuple as a single
        # positional argument: func(unwrap(args), ...). For functions like
        # torch.nn.functional.linear(input, weight, bias) this meant 'input'
        # became a tuple (input, weight, bias) instead of a Tensor, raising:
        #   TypeError: linear(): argument 'input' must be Tensor, not tuple
        # We fix this by splatting the arguments.
        unwrapped_args = unwrap(args)
        if not isinstance(unwrapped_args, (list, tuple)):
            # safety: ensure we always have an iterable for * expansion
            unwrapped_args = [unwrapped_args]
        out_real = func(*unwrapped_args, **unwrap(kwargs))
    # If not capturing or the output contains no Tensor, return the raw result (no node)
        def any_tensor(x):
            if isinstance(x, torch.Tensor):
                return True
            if isinstance(x, (list, tuple)):
                return any(any_tensor(e) for e in x)
            if isinstance(x, dict):
                return any(any_tensor(v) for v in x.values())
            return False
        if not capturing:
            # Module-level: don't record ops, but keep Wrapper flowing to maintain inter-module chain
            def inherit_last(xs):
                for w in xs:
                    if getattr(w, 'last_node', None) is not None:
                        return w.last_node
                return None
            if any_tensor(out_real):
                inherited = inherit_last(inputs)
                if inherited is None:
                    return out_real
                def wrap_passthrough(y):
                    if isinstance(y, torch.Tensor):
                        w = TensorWrapper(y, self.graph)
                        w.last_node = inherited
                        return w
                    if isinstance(y, (list, tuple)):
                        return type(y)(wrap_passthrough(e) for e in y)
                    return y
                return wrap_passthrough(out_real)
            return out_real
    # Capturing: build node, connect edges from inputs, and record timing
        self.graph.add_opr_node(func.__name__, tuple(
            w.last_node.id for w in inputs if w.last_node is not None
        ))
        new_node = self.graph.nodes[-1]
        added = set()
        for w in inputs:
            if getattr(w, 'last_node', None) is not None and w.last_node.id not in added:
                self.graph.add_edge(w.last_node, new_node)
                added.add(w.last_node.id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_ms = time.time() * 1000.0
        new_node.forward_time += (end_ms - (start_ms or end_ms))
        new_node.forward_epoch += 1

        # 4) wrap outputs back, reusing our graph and stamping last_node=new_node
        def wrap_out(y):
            if isinstance(y, torch.Tensor):
                w = TensorWrapper(y, self.graph)
                if new_node is not None:
                    w.last_node = new_node
                return w
            if isinstance(y, (list, tuple)):
                return type(y)(wrap_out(e) for e in y)
            return y

        return wrap_out(out_real)

    # helpers for op timing
    def _op_start(self, name, args):
        if self.graph is None or self.graph.is_freeze or not getattr(self.graph, 'capture_ops', True):
            return None
        self.graph.add_opr_node(name, args)
        node = self.graph.nodes[-1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        node._start_time_ms = time.time() * 1000.0
        return node

    def _op_end(self, node):
        if node is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_ms = time.time() * 1000.0
    # Tolerance: _start_time_ms might be missing (doesn't affect module-level)
        start_ms = getattr(node, '_start_time_ms', end_ms)
        node.forward_time += (end_ms - start_ms)
        node.forward_epoch += 1
    # ===== end key part =====