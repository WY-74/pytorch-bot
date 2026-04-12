import os
import nltk
import random
import torch
from torch import nn
from utils import base
from utils.block import attention_tools


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _read_wiki():
    file_name = os.path.join("/root/autodl-tmp/d2l/dataset/wikitext-2", "wiki.train.tokens")
    with open(file_name, 'r') as f:
        lines = f.readlines()

    paragraphs = [
        nltk.tokenize.sent_tokenize(line.strip().lower())
        for line in lines
        if len(nltk.tokenize.sent_tokenize(line)) >= 2
    ]
    # paragraphs = [line.strip().lower().split(' . ') for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False

    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))

    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []

    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break

        masked_token = None
        if random.random() < 0.8:
            masked_token = '<mask>'  # 80%的时间：将词替换为“<mask>”词元
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]  # 10%的时间：保持词不变
            else:
                masked_token = random.choice(vocab.idx_to_token)  # 10%的时间：用随机词替换该词

        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))

    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds = max(1, round(len(tokens) * 0.15))  # 应该抽多少token

    # mlm_input_tokens: 替换15%词元后的tokens
    # pred_positions_and_labels: List[Tuple[替换位置索引, 原词元]]
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab
    )
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])

    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    (
        all_token_ids,
        all_segments,
        valid_lens,
    ) = (
        [],
        [],
        [],
    )
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []

    for token_ids, pred_positions, mlm_pred_label_ids, segments, is_next in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab["pad"]] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(
            torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long)
        )
        all_mlm_weights.append(
            torch.tensor(
                [1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32
            )
        )
        all_mlm_labels.append(
            torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long)
        )
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))

    return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)


def load_data_wiki(batch_size, max_len):
    num_workers = os.cpu_count() // 2
    paragraphs = _read_wiki()
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)

    return train_iter, train_set.vocab


class BERTEncoder(nn.Module):
    """BERT编码器"""

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        max_len=1000,
        query_size=768,
        key_size=768,
        value_size=768,
        **kwargs,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"{i}",
                attention_tools.EncoderBlock(
                    query_size,
                    key_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    use_bias=True,
                ),
            )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, : X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""

    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens), nn.ReLU(), nn.LayerNorm(num_hiddens), nn.Linear(num_hiddens, vocab_size)
        )

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)

        masked_X = X[batch_idx, pred_positions]  # torch.Size([NumPositions, NumHiddens])
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""

    def __init__(self, num_inputs=768, **kwargs):
        super().__init__()
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)


class BERTModel(nn.Module):
    """BERT模型"""

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        max_len=1000,
        key_size=768,
        query_size=768,
        value_size=768,
        hid_in_features=768,
        mlm_in_features=768,
        nsp_in_features=768,
    ):
        super().__init__()
        self.encoder = BERTEncoder(
            vocab_size,
            num_hiddens,
            norm_shape,
            ffn_num_input,
            ffn_num_hiddens,
            num_heads,
            num_layers,
            dropout,
            max_len=max_len,
            query_size=query_size,
            key_size=key_size,
            value_size=value_size,
        )
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        paragraphs = [base.tokenize(paragraph, token_type="word") for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = base.Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])

        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            # List[Tuple[<cls>xxx<sep>xxx<sep>, segments, is_next]]
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
        # examples: List[Tuple[替换词元后tokens, 被替换位置索引, 被替换的原词元, segments, is_next]]
        examples = [
            (_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
            for tokens, segments, is_next in examples
        ]

        # 填充输入
        (
            self.all_token_ids,
            self.all_segments,
            self.valid_lens,
            self.all_pred_positions,
            self.all_mlm_weights,
            self.all_mlm_labels,
            self.nsp_labels,
        ) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (
            self.all_token_ids[idx],
            self.all_segments[idx],
            self.valid_lens[idx],
            self.all_pred_positions[idx],
            self.all_mlm_weights[idx],
            self.all_mlm_labels[idx],
            self.nsp_labels[idx],
        )

    def __len__(self):
        return len(self.all_token_ids)
