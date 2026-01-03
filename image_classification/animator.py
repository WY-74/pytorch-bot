from typing import Tuple, List
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib_inline import backend_inline
from IPython import display


class Animator:
    def __init__(
        self,
        nrows: int = 1, ncols: int = 1,
        figsize: Tuple[int|float, int|float]|None = None,
        fmts: Tuple[str, ...] = ("-", "m--", "g-.", "r:"),
        xlabel: str|None = None, ylabel: str|None = None,
        xlim: Tuple[float|None, float|None] = (None, None), ylim: Tuple[float|None, float|None] = (None, None),
        xscale: str = "linear", yscale: str = "linear",
        legend: List[str]|None = None,
    ):
        self.fig, self.axes = self._init_figsize(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self._set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def _init_figsize(self, nrows: int = 1, ncols: int = 1, figsize: Tuple[int|float, int|float]|None = None):
        if figsize is None:
            figsize = (3.5, 2.5)
        backend_inline.set_matplotlib_formats('svg')
        return plt.subplots(nrows, ncols, figsize=figsize)
     
    def _set_axes(
            self,
            axes: Axes,
            xlabel: str|None = None, ylabel: str|None = None,
            xlim: Tuple[float|None, float|None] = (None, None), ylim: Tuple[float|None, float|None] = (None, None),
            xscale: str = "linear", yscale: str = "linear",
            legend: List[str]|None = None,
        ):
        if xlabel:
            axes.set_xlabel(xlabel)
        if ylabel:
            axes.set_ylabel(ylabel)
        axes.set_xlim(*xlim)
        axes.set_ylim(*ylim)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        if legend:
            axes.legend(legend)
        axes.grid()
    
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