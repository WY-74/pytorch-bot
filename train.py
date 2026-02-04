import time
import torch
from torch import nn
from IPython import display
from typing import Tuple, List
from matplotlib.axes import Axes


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


class Animator:
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Tuple[int | float, int | float] | None = None,
        fmts: Tuple[str, ...] = ("-", "m--", "g-.", "r:"),
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: Tuple[float | None, float | None] = (None, None),
        ylim: Tuple[float | None, float | None] = (None, None),
        xscale: str = "linear",
        yscale: str = "linear",
        legend: List[str] | None = None,
    ):
        self.fig, self.axes = init_figsize(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        display.display(self.fig)
        display.clear_output(wait=True)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def __getitem__(self, idx):
        return self.data[idx]

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)


class Train:
    def __init__(self, net, loss, optimer):
        self.net = self._init_net(net)
        self.loss = loss
        self.optimer = optimer

    def _try_all_gpus(self):
        """返回所有可用的GPU，如果没有GPU，则返回[]"""
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return None
        return [torch.device(f'cuda:{i}') for i in range(device_count)]

    def _init_net(self, net):
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

    def _grad_clipping(self, theta):
        if isinstance(self.net, nn.Module):
            params = [p for p in self.net.parameters() if p.requires_grad]
        else:
            params = self.net.params

        norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    def run(
        self,
        train_iter,
        num_epochs,
        test_iter=None,
        grad_clipping: bool = True,
        grad_clipping_theta: int = 1,
        metric: None | Accumulator = None,
        animator: None | Animator = None,
        *args,
        **kwargs,
    ):
        self.timer = Timer()
        for epoch in range(num_epochs):
            self.timer.start()
            args, kwargs = self.init_state(*args, **kwargs)
            for i, (X, y) in enumerate(train_iter):
                self.net.train()
                self.optimer.zero_grad()
                if self.devices is not None:
                    X = X.to(self.devices[0], non_blocking=True)
                    y = y.to(self.devices[0], non_blocking=True)

                l, args, kwargs = self.trainer(X, y, *args, **kwargs)
                if l is None:
                    raise ValueError("损失不可以为None")
                l.backward()
                if grad_clipping:
                    self._grad_clipping(grad_clipping_theta)

                self.optimer.step()
                with torch.no_grad():
                    self.update_metric(metric, epoch, l, y.numel())

            self.timer.stop()
            self.update_animator(animator, metric, epoch)
        self.predict(test_iter, *args, **kwargs)
        self.print_results(metric)

    def init_state(self, *args, **kwargs):
        """必要时初始化网络状态"""
        return args, kwargs

    def trainer(self, X, y, *args, **kwargs):
        """训练"""
        return l, args, kwargs

    def update_metric(self, metric, epoch, l, numel):
        return None

    def predict(self, test_iter, *args, **kwargs):
        return None

    def update_animator(self, animator, metric, epoch):
        return None

    def print_results(self, metric):
        return None
