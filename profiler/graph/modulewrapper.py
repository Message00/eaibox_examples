import torch.nn as nn
from .tensorwrapper import TensorWrapper

class ModuleWrapper(nn.Module):
    def __init__(self, module):
        super(ModuleWrapper, self).__init__()
        self.module = module
        
    def forward(self, x: TensorWrapper):
        result_tensor = self.module(x.tensor)
        return TensorWrapper(result_tensor)