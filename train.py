import torch
from torch import nn


class Train:
    def __init__(self, net, loss, optimizer, jit_script: bool = False):
        self.net = self._init_net(net, jit_script)
        self.loss = loss
        self.optimizer = optimizer

    def _try_all_gpus(self):
        """返回所有可用的GPU，如果没有GPU，则返回[]"""
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return None
        return [torch.device(f'cuda:{i}') for i in range(device_count)]

    def _init_net(self, net, jit_script: bool = False):
        net.apply(self.init_weights)
        if jit_script:
            net = torch.jit.script(net)

        self.devices = self._try_all_gpus()
        if self.devices is not None:
            if len(self.devices) > 1:
                net = net.to(self.devices[0])
                net = nn.DataParallel(net, device_ids=self.devices)
            else:
                net = net.to(self.devices[0])
        else:
            self.devices = ["cpu"]
        return net

    def grad_clipping(self, theta):
        if isinstance(self.net, nn.Module):
            params = [p for p in self.net.parameters() if p.requires_grad]
        else:
            params = self.net.params

        norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    def init_weights(self, m):
        return

    def train_epochs(self, *args, **kwargs):
        return
