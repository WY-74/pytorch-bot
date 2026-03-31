import math
import os
import random
import torch
from torch import nn
from torch.utils import data
from utils import base
from utils.train import Train


def read_ptb():
    data_dir = "/root/autodl-tmp/d2l/dataset/ptb/ptb.train.txt"
    with open(data_dir) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split("\n")]


def subsample(sentences, vocab):
    """下采样高频词"""
    counter = base.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return random.uniform(0, 1) < math.sqrt(1e-4 / (counter[token] / num_tokens))

    return [[token for token in line if vocab[token] != vocab.unk and keep(token)] for line in sentences], counter


def get_centers_and_contexts(corpus, window_size):
    centers, contexts = [], []
    for line in corpus:
        # 要形成“中心词-上下文词”对，每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            indices = list(range(max(0, i - window_size), min(len(line), i + window_size + 1)))
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])

    return centers, contexts


class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""

    def __init__(self, sampling_weights):
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词"""
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]  # 不包含 "<unk>"
    all_negatives, generator = [], RandomGenerator(sampling_weights)

    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)

    return all_negatives


def batchify(data):
    """
    返回带有负采样的跳元模型的小批量样本
    ARGS:
      - data: ([center, context, negative], ...)
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers.append(center)
        contexts_negatives.append(context + negative + [0] * (max_len - cur_len))
        masks.append([1] * cur_len + [0] * (max_len - cur_len))
        labels.append([1] * len(context) + [0] * (max_len - len(context)))

    return (
        torch.tensor(centers).reshape((-1, 1)),
        torch.tensor(contexts_negatives),
        torch.tensor(masks),
        torch.tensor(labels),
    )


def load_data_ptb(batch_size, window_size, num_noise_words):
    """下载PTB数据集，然后将其加载到内存中"""
    sentences = read_ptb()
    vocab = base.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, window_size=window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, idx):
            return (
                self.centers[idx],
                random.choices(self.contexts[idx], k=window_size // 2),
                random.choices(self.negatives[idx], k=window_size // 2 * num_noise_words),
            )

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify, num_workers=5)
    return data_iter, vocab


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


def train_word2vec(net, data_iter, lr, num_epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = SigmoidBCELoss()
    TrainWord2Vec(net, loss, optimizer).train_epochs(net, data_iter, num_epochs)


class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(inputs, target, weight=mask, reduction="none")
        if mask is not None:
            return out.sum(dim=1) / mask.sum(dim=1)
        return out.mean(dim=1)


class TrainWord2Vec(Train):
    def init_weights(self, m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    def train_epochs(self, net, data_iter, num_epochs, *args, **kwargs):
        super().train_epochs(*args, **kwargs)
        animator = base.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs])
        metric = base.Accumulator(2)
        for epoch in range(num_epochs):
            timer, num_batches = base.Timer(), len(data_iter)
            for i, batch in enumerate(data_iter):
                self.optimizer.zero_grad()
                center, context_negative, mask, label = [data.to(self.device) for data in batch]
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = self.loss(pred.reshape(label.shape).float(), label.float(), mask)
                l.sum().backward()
                self.optimizer.step()
                metric.add(l.sum(), l.numel())
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1],))
        print(f'loss {metric[0] / metric[1]:.3f}', f'{metric[1] / timer.stop():.1f} tokens/sec on {str(self.device)}')
