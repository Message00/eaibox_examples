import torch.nn as nn

def torch_function_wrapper(func, graph, *args):
    # TODO: Implement this function
    # This function should wrap the torch function with a Node object
    # and add it to the graph.
    # The function should return the output of the torch function.
    # The aim of this function is to profile the torch functions in layerwise profiling.
    pass

class Node(object):
    def __init__(self, graph, id: int, description: str, module: type, args: dict, **kwargs):
        super(Node, self).__init__()
        self.graph = graph
        self.module = module
        self.id = id
        self.forward_epoch = 0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.param_size = -1
        self.description = description
        self.args = args
        
    