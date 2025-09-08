import torch
import torch.nn as nn
from .tensorwrapper import TensorWrapper
from .node import Node

class Edge:
    def __init__(self, source: Node, target: Node) -> None:
        self.source = source
        self.target = target

class Graph:
    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.counter = 0
        self.is_freeze = False
    # Whether to capture operator-level nodes (Add/View/Reshape, etc.).
    # If False, only keep the module-level node chain.
        self.capture_ops = True

    def freeze(self):
        self.is_freeze = True
        
    def add_opr_node(self, description: str, args) -> int:
        if not self.is_freeze:
            node = Node(self, self.get_new_id(), description, None, args)
            self.nodes.append(node)
            return node.id
        else:
            return -1
    
    def add_nn_node(self, node: Node):
        if self.is_freeze:
            return
        self.nodes.append(node)
        
    def add_edge(self, source: Node, target: Node):
        if self.is_freeze:
            return
        edge = Edge(source, target)
        self.edges.append(edge)

    def get_node(self, id: int) -> Node:
        for node in self.nodes:
            if node.id == id:
                return node
        return None

    def get_new_id(self) -> int:
        self.counter += 1
        return self.counter
    
    def summary(self):
        print("NodeID\tDescription\tArgs\tForwardTime\tParamSize")
        for node in self.nodes:
            if node.description in ["Add", "Mul", "Add(inplace)"]:
                if isinstance(node.args[0], torch.Tensor):
                    print(node.id, "\t", node.description, "\t", f"Tensor + output {node.args[1]}", "\t", node.forward_time)
                else:
                    print(node.id, "\t", node.description, "\t", f"output {node.args[0]} + output {node.args[1]}", "\t", node.forward_time)
            elif node.forward_epoch > 0:
                print(node.id, "\t", node.description, "\t", node.args, "\t", node.forward_time / node.forward_epoch, "\t", node.param_size)
            else:
                print(node.id, "\t", node.description, "\t", node.args, "\t", node.forward_time, "\t", node.param_size)
        for edge in self.edges:
            print(edge.source.id, " -> ", edge.target.id)

    def stats(self):
        """Print graph statistics: totals, module/op counts, edge count, and top module forward times."""
        nn_nodes = [n for n in self.nodes if n.module is not None]
        op_nodes = [n for n in self.nodes if n.module is None]
        print("==== Graph Stats ====")
        print(f"Total nodes: {len(self.nodes)}")
        print(f"  Module nodes: {len(nn_nodes)}")
        print(f"  Op nodes: {len(op_nodes)}")
        print(f"Total edges: {len(self.edges)}")
        time_nodes = [n for n in nn_nodes if n.forward_time > 0]
        time_nodes.sort(key=lambda x: x.forward_time, reverse=True)
        if time_nodes:
            print("Top 5 module forward times (ms total over epochs):")
            for n in time_nodes[:5]:
                avg = n.forward_time / max(1, n.forward_epoch)
                print(f"  id={n.id:<4} {n.description:<25} total={n.forward_time:.2f} avg={avg:.2f} epochs={n.forward_epoch}")
        print("=====================")

    def to_text(self, include_edges: bool = True, include_stats: bool = True) -> str:
        lines = []
        if include_stats:
            nn_nodes = [n for n in self.nodes if n.module is not None]
            op_nodes = [n for n in self.nodes if n.module is None]
            lines.append("==== Stats ====")
            lines.append(f"Total nodes: {len(self.nodes)} (modules={len(nn_nodes)}, ops={len(op_nodes)})")
            lines.append(f"Total edges: {len(self.edges)}")
            lines.append("================")
        lines.append("NodeID\tType\tAvgForward(ms)\tBackward(ms)\tParamSize\tDesc\tGPUMemory(MB)")
        for node in self.nodes:
            if node.forward_epoch > 0:
                avg = node.forward_time / node.forward_epoch
            else:
                avg = 0.0
            gpu_mem = 0.0 if node.param_size <= 0 else node.param_size * 2 / 1024 / 1024
            ntype = "Module" if node.module is not None else "Op"
            lines.append(f"{node.id}\t{ntype}\t{avg:.3f}\t\t{node.backward_time}\t\t{node.param_size}\t{node.description}\t{gpu_mem:.3f}")
        if include_edges:
            lines.append("-- Edges --")
            if self.edges:
                for e in self.edges:
                    lines.append(f"{e.source.id} -> {e.target.id}")
            else:
                # Fallback: if no explicit edges recorded, chain module nodes by order for readability
                nn_nodes = [n for n in self.nodes if n.module is not None]
                for a, b in zip(nn_nodes, nn_nodes[1:]):
                    lines.append(f"{a.id} -> {b.id}")
        return "\n".join(lines)

    def export(self, filepath: str, include_edges: bool = True, include_stats: bool = True):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_text(include_edges=include_edges, include_stats=include_stats))

    def get_total_forward_time(self):
        total_time = 0
        for node in self.nodes:
            total_time += node.forward_time
        return total_time
    
    def get_total_backward_time(self):
        total_time = 0
        for node in self.nodes:
            total_time += node.backward_time
        return total_time
    
    def get_forward_time_list(self):
        time_list = []
        for node in self.nodes:
            time_list.append(node.forward_time)
        return time_list
