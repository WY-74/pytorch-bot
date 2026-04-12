import re
import os
import spacy
import torch
from utils import base


def tokenizer(lines, token_type='word'):
    if token_type == 'word':
        return [line.split() for line in lines]
    if token_type == "char":
        return [list(line) for line in lines]
    print('错误：未知词元类型：' + token)


def tokenizer_with_spacy(lines):
    tokens = []
    spacy_en = spacy.load("en_core_web_sm")
    for line in lines:
        tokens.append([tok.text for tok in spacy_en.tokenizer(line)])
    return tokens


def load_data_imdb(batch_size, num_steps=500):
    def _read_imdb(is_train):
        """读取IMDb评论数据集文本序列和标签"""
        data, labels = [], []
        data_dir = "/root/autodl-tmp/d2l/dataset/aclImdb"

        for label in ('pos', 'neg'):
            folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
            for file in os.listdir(folder_name):
                with open(os.path.join(folder_name, file), encoding='utf-8') as f:
                    review = f.read().replace("\n", "")
                    data.append(review)
                    labels.append(1 if label == 'pos' else 0)

        return data, labels

    # Train
    train_data = _read_imdb(is_train=True)
    train_tokens = tokenizer_with_spacy(train_data[0])
    vocab = base.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    train_features = torch.tensor([base.truncate_pad(vocab[line], num_steps, vocab["pad"]) for line in train_tokens])
    train_iter = base.load_array([train_features, torch.tensor(train_data[1])], batch_size=64)

    # Test
    test_data = _read_imdb(is_train=False)
    test_tokens = tokenizer_with_spacy(test_data[0])
    test_features = torch.tensor([base.truncate_pad(vocab[line], num_steps, vocab["pad"]) for line in test_tokens])
    test_iter = base.load_array([test_features, torch.tensor(test_data[1])], batch_size=64)

    return train_iter, test_iter, vocab


def load_data_snli(batch_size, num_steps=50):
    def _extract_text(s):
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    def _read_snli(is_train):
        label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        file_name = os.path.join(
            "/root/autodl-tmp/d2l/dataset/snli_1.0", 'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt'
        )
        with open(file_name, 'r') as f:
            rows = [row.split('\t') for row in f.readlines()[1:]]

        premises, hypotheses, labels = [], [], []
        for row in rows:
            if row[0] not in label_set:
                continue
            premises.append(_extract_text(row[1]))
            hypotheses.append(_extract_text(row[2]))
            labels.append(label_set[row[0]])
        return premises, hypotheses, labels

    class SNLIDataset(torch.utils.data.Dataset):
        """用于加载SNLI数据集的自定义数据集"""

        def __init__(self, dataset, num_steps, vocab=None):
            self.num_steps = num_steps
            all_premise_tokens = tokenizer(dataset[0])
            all_hypothesis_tokens = tokenizer(dataset[1])
            if vocab is None:
                self.vocab = base.Vocab(
                    all_premise_tokens + all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>']
                )
            else:
                self.vocab = vocab

            self.premises = self._pad(all_premise_tokens)
            self.hypotheses = self._pad(all_hypothesis_tokens)
            self.labels = torch.tensor(dataset[2])
            print(f"read {len(self.premises)} examples")

        def __getitem__(self, idx):
            return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

        def __len__(self):
            return len(self.premises)

        def _pad(self, lines):
            return torch.tensor(
                [base.truncate_pad(self.vocab[line], self.num_steps, self.vocab["<pad>"]) for line in lines]
            )

    num_workers = os.cpu_count() // 2

    # Train
    train_data = _read_snli(is_train=True)
    train_set = SNLIDataset(train_data, num_steps)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)

    # Test
    test_data = _read_snli(is_train=False)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter, train_set.vocab
