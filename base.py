import re
import math
import time
import torch
import random
import collections
import numpy as np

from torch import nn
from torch import Tensor
from torch.nn import functional as F
from PIL import Image
from IPython import display
from typing import Tuple, List, Optional
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib_inline import backend_inline

from utils.train import Train


def init_figsize(nrows: int = 1, ncols: int = 1, figsize: Tuple[int | float, int | float] | None = None):
    if figsize is None:
        figsize = (3.5, 2.5)
    backend_inline.set_matplotlib_formats('svg')
    return plt.subplots(nrows, ncols, figsize=figsize)


def set_figsize(figsize: tuple = (3.5, 2.5)):
    """设置matplotlib的图表大小"""
    backend_inline.set_matplotlib_formats('svg')  # 使用svg格式在Jupyter中显示绘图
    plt.rcParams['figure.figsize'] = figsize


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def plot(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=[],
    xlim=None,
    ylim=None,
    xscale='linear',
    yscale='linear',
    fmts=('-', 'm--', 'g-.', 'r:'),
    figsize=(3.5, 2.5),
    axes=None,
):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__")

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_utils`"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset.

    Defined in :numref:`sec_utils`"""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[]"""
    device_count = torch.cuda.device_count()
    if device_count == 0:
        return None
    return [torch.device(f'cuda:{i}') for i in range(device_count)]


def accuracy(y_hat, y) -> float:
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[-1] > 1:
        # 取矩阵每列最大值(每条数据最大概率的类别)，返回一个batchsize长度一维数组作为预测类别
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)  # 正确预测的数量，总预测的数量
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(metric.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 语言模型
def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    if token == "char":
        return [list(line) for line in lines]
    print('错误：未知词元类型：' + token)


def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        _tokens = []
        for line in tokens:
            for token in line:
                _tokens.append(token)
        tokens = _tokens
    return collections.Counter(tokens)


def read_timemachine():
    with open("data/timemachine.txt") as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def load_corpus_time_machine(max_tokens=-1):
    lines = read_timemachine()
    tokens = tokenize(lines, token="char")
    vocab = Vocab(tokens)

    corpus = []
    for line in lines:
        for token in line:
            corpus.append(vocab[token])

    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def predict_ch8(prefix, num_preds, net, vocab):
    device = next(net.parameters()).device
    state = None
    # state = net.begin_state(batch_size=1, device=device)

    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        logits, state = net(get_input(), state)
        outputs.append(logits.argmax(dim=1).reshape(1))

    return ''.join([vocab.idx_to_token[i] for i in outputs])


# 训练调用函数
def train_ch8(net, train_iter, num_epochs, vocab, lr, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        optimer = torch.optim.SGD(net.parameters(), lr)
    else:
        optimer = lambda batch_size: sgd(net.params, lr, batch_size)
    metric = Accumulator(2)
    animator = Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab)

    TrainCh8(net, loss, optimer).run(
        train_iter,
        num_epochs,
        test_iter="time traveller",
        metric=metric,
        animator=animator,
        use_random_iter=use_random_iter,
        vocab=vocab,
    )

    print(predict('time traveller'))
    print(predict('traveller'))


# 训练器
class TrainCh8(Train):
    def init_state(self, *args, **kwargs):
        kwargs["state"] = None
        return args, kwargs

    def trainer(self, X, y, *args, **kwargs):
        state = kwargs["state"]
        use_random_iter = kwargs["use_random_iter"]

        if use_random_iter:
            state = None

        y = y.T.reshape(-1)
        y_hat, new_state = self.net(X, state)

        l = self.loss(y_hat, y.long())

        if new_state is not None:
            if isinstance(new_state, tuple):
                new_state = (new_state[0].detach(), new_state[1].detach())
            else:
                new_state = new_state.detach()
        kwargs["state"] = new_state

        return l, args, kwargs

    def update_metric(self, metric, epoch, l, numel):
        metric.add(l * numel, numel)

    def predict(self, test_iter, *args, **kwargs):
        vocab = kwargs["vocab"]

        state = None
        outputs = [vocab[test_iter[0]]]
        get_input = lambda: torch.tensor([outputs[-1]], device=self.devices[0]).reshape((1, 1))
        for y in test_iter[1:]:
            _, state = self.net(get_input(), state)
            outputs.append(vocab[y])
        for _ in range(50):
            logits, state = self.net(get_input(), state)
            outputs.append(logits.argmax(dim=1).reshape(1))

        print(''.join([vocab.idx_to_token[i] for i in outputs]))

    def update_animator(self, animator, metric, epoch):
        ppl = math.exp(metric[0] / metric[1])
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])

    def print_results(self, metric):
        ppl = math.exp(metric[0] / metric[1])
        speed = metric[1] / self.timer.sum()
        print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(self.devices)}')


# 常用 Model
class RNN(nn.Module):
    def __init__(self, rnn_layer, vocab_size, num_hiddens, **kwargs):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(num_hiddens, vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state: Optional[Tensor] = None):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))  # input: torch.Size([num_steps*batch_size, vocab_size]);
        return output, state


class LSTM(nn.Module):
    def __init__(self, rnn_layer, vocab_size, num_hiddens, **kwargs):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(num_hiddens, vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state: Optional[Tuple[Tensor, Tensor]] = None):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))  # input: torch.Size([num_steps*batch_size, vocab_size]);
        return output, state


# 其余常用类
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (tuple, list)):
            return self.idx_to_token[indices]
        return [self.to_tokens(index) for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


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


class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = self._seq_data_iter_random
        else:
            self.data_iter_fn = self._seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

    def _seq_data_iter_random(self, corpus, batch_size, num_steps):
        corpus = corpus[random.randint(0, num_steps - 1) :]
        num_subseqs = (len(corpus) - 1) // num_steps  # 减去1，是因为我们需要考虑标签

        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        random.shuffle(initial_indices)

        def data(pos):
            return corpus[pos : pos + num_steps]

        num_batches = num_subseqs // batch_size
        for i in range(0, num_batches * batch_size, batch_size):
            initial_indices_per_batch = initial_indices[i : i + batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            Y = [data(j + 1) for j in initial_indices_per_batch]
            yield torch.tensor(X), torch.tensor(Y)

    def _seq_data_iter_sequential(self, corpus, batch_size, num_steps):
        """
        使用顺序分区生成一个小批量子序列
        """
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = torch.tensor(corpus[offset : offset + num_tokens])
        Ys = torch.tensor(corpus[offset + 1 : offset + 1 + num_tokens])
        Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        num_batches = Xs.shape[1] // num_steps
        for i in range(0, num_steps * num_batches, num_steps):
            X = Xs[:, i : i + num_steps]
            Y = Ys[:, i : i + num_steps]
            yield X, Y
