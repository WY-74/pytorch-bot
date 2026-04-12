import math
import torch
from torch import nn
from typing import List
from utils import base


def sequence_mask(X, valid_len, value=0):
    """
    在序列中屏蔽不相关的项
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """
    Args:
      - X: torch.Size([BatchSize, NumQuerySteps, NumKeySteps])
      - valid_lens: torch.Size([BatchSize])
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])  # torch.Size([BatchSize*NumQuerySteps])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

    return nn.functional.softmax(X.reshape(shape), dim=-1)


def transpose_qkv(X, num_heads, remove_heads_idx: None | int = None):
    """
    为了多注意力头的并行计算而变换形状
    Args:
      - X: torch.Size([BatchSize, NumQueries or NumKeys or NumValues, NumHiddens])
    Return:
      - torch.Size([BatchSize*NumHeads, NumQueries or NumKeys or NumValues, NumHiddens])
    """
    batch_size = X.shape[0]

    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)  # torch.Size([~, ~, NumHeads, NumHiddens/NumHeads])
    X = X.permute(0, 2, 1, 3)
    X = X.reshape(-1, X.shape[2], X.shape[3])

    if remove_heads_idx is not None:
        remove_positions = [i * num_heads + remove_heads_idx for i in range(batch_size)]
        _mask = torch.ones(X.size(0), dtype=torch.bool)
        _mask[remove_positions] = False
        X = X[_mask, :, :]
        return X, remove_positions

    return X, None


def transpose_output(X, num_heads, insert_positions: None | List[int] = None):
    """
    逆转 transpose_qkv 函数
    Args:
      - X: torch.Szie([BatchSize*NumHeads, NumQueries, NumHiddens/NumHeads])
    Return:
      - torch.Size([BatchSize, NumQueries, NumHiddens])
    """
    if insert_positions is not None:
        batch_size = int(X.shape[0] / (num_heads - 1))
        target_batch_size = X.shape[0] + batch_size
        target = torch.zeros(target_batch_size, *X.shape[1:], device=X.device)
        target_indices = torch.arange(target_batch_size)

        insert_mask = torch.zeros(target_batch_size, dtype=torch.bool)
        insert_mask[insert_positions] = True

        non_insert_indices = target_indices[~insert_mask]
        target[non_insert_indices] = X
    else:
        target = X

    target = target.reshape(-1, num_heads, target.shape[1], target.shape[2])
    target = target.permute(0, 2, 1, 3)
    target = target.reshape(target.shape[0], target.shape[1], -1)

    return target


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)  # 索引偶数列
        self.P[:, :, 1::2] = torch.cos(X)  # 索引奇数列

    def forward(self, X, step: int | None = None):
        if step is None:
            X = X + self.P[:, : X.shape[1], :].to(X.device)
        else:
            X = X + self.P[:, step : step + 1, :].to(X.device)
        return self.dropout(X)


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.drop(Y) + X)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


# Attention
class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)  # features: torch.Size([BatchSize, NumQueries, NumKVs, NumHiddens])

        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Args:
          - queries: tprch.Size([BatchSize, NumSteps, NumHiddens])
          - keys: tprch.Size([BatchSize, NumSteps, NumHiddens])
          - values: tprch.Size([BatchSize, NumSteps, NumHiddens])
          - valid_lens: tprch.Size([BatchSize])
        Return:
          - torch.Size([BatchSize, NumQuerySteps, NumHiddens])
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(
            d
        )  # torch.Size([BatchSize, NumQuerySteps, NumKeySteps])
        self.attention_weights = masked_softmax(
            scores, valid_lens
        )  # torch.Size([BatchSize, NumQuerySteps, NumKeySteps])
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.core_attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens, remove_heads_idx: None | int = None):
        """
        Args:
          - queries: tprch.Size([BatchSize, NumSteps, num_hiddens])
          - keys: tprch.Size([BatchSize, NumSteps, num_hiddens])
          - values: tprch.Size([BatchSize, NumSteps, num_hiddens])
          - valid_lens: tprch.Size([BatchSize])
        Return:
          - torch.Size([BatchSize, NumQuerySteps, num_hiddens])
        """
        queries, remove_positions = transpose_qkv(
            self.W_q(queries), self.num_heads, remove_heads_idx
        )  # torch.Size([BatchSize*num_heads, NumSteps, num_hiddens/num_heads])
        keys, _ = transpose_qkv(
            self.W_k(keys), self.num_heads, remove_heads_idx
        )  # torch.Size([BatchSize*num_heads, NumSteps, num_hiddens/num_heads])
        values, _ = transpose_qkv(
            self.W_v(values), self.num_heads, remove_heads_idx
        )  # torch.Size([BatchSize*num_heads, NumSteps, num_hiddens/num_heads])

        if valid_lens is not None:
            if remove_positions is not None:
                valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads - 1, dim=0)
            else:
                valid_lens = torch.repeat_interleave(
                    valid_lens, repeats=self.num_heads, dim=0
                )  # torch.Size([BatchSize*num_heads])

        output = self.core_attention(
            queries, keys, values, valid_lens
        )  # torch.Size([BatchSize*num_heads, NumQuerySteps, num_hiddens/num_heads])
        output_concat = transpose_output(
            output, self.num_heads, remove_positions
        )  # torch.Size([BatchSize, NumQuerySteps, num_hiddens]), 合并多头

        return self.W_o(output_concat)


# Block
class EncoderBlock(nn.Module):
    """Transformer编码块"""

    def __init__(
        self,
        query_size,
        key_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        """
        Args:
          - X: torch.Size([BatchSize, NumSteps, num_hiddens]) 填加了位置信息的数据
          - valid_lens: torch.Size([BatchSize])
        Return:
          - torch.Size([BatchSize, NumSteps, num_hiddens])
        """
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))  # torch.Size([BatchSize, NumSteps, num_hiddens])
        return self.addnrom2(Y, self.ffn(Y))


class DecoderBlock(nn.Module):
    """Transformer解码块"""

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        i,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X  # torch.Size([BatchSize, NumSteps, num_hiddens])
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(
                batch_size, 1
            )  # torch.Size([BatchSize, NumSteps])
        else:
            dec_valid_lens = None

        X2 = self.attention1(
            X, key_values, key_values, dec_valid_lens
        )  # torch.Size([BatchSize, NumSteps, num_hiddens]), 每一个step对当前step和之前step的关联程度
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(
            Y, enc_outputs, enc_outputs, enc_valid_lens
        )  # torch.Size([BatchSize, NumSteps, num_hiddens]), 每一个step对编码信息的注意力
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


# Encoder
class TransformerEncoder(base.Encoder):
    """Transformer编码器"""

    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"block{str(i)}",
                EncoderBlock(
                    query_size,
                    key_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    use_bias,
                ),
            )

    def forward(self, X, valid_lens, *args):
        """
        Args:
          - X: torch.Size([BatchSize, NumSteps])
          - valid_lens: torch.Size([BatchSize])
        return:
          - torch.Size([BatchSize, NumSteps, num_hiddens])
        """
        X = self.pos_encoding(
            self.embedding(X) * math.sqrt(self.num_hiddens)
        )  # torch.Size([BatchSize, NumSteps, num_hiddens])
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)  # torch.Size([BatchSize, NumSteps, num_hiddens])
            self.attention_weights[i] = blk.attention.core_attention.attention_weights
        return X


# Decoder
class AttentionDecoder(base.Decoder):
    """带有注意力机制解码器的基本接口"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.rnn = nn.LSTM(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, outputs, enc_valid_lens, remove_heads_idx: None | int = None, *args):
        """
        Args:
          - outputs: Tuple[enc_output, enc_state]
              - enc_outputs: torch.Size([NumSteps, BatchSize, NumHiddens])
              - enc_hidden_state:
                - GRU: torch.Size([num_layers, BatchSize, NumHiddens])
                - LSTM: Tuple[torch.Size([num_layers, BatchSize, NumHiddens]), torch.Size([num_layers, BatchSize, NumHiddens])]
          - enc_valid_lens: torch.Size([BatchSize])
        """
        enc_outputs, enc_hidden_state = outputs
        return (enc_outputs.permute(1, 0, 2), enc_hidden_state, enc_valid_lens, remove_heads_idx)

    def forward(self, X, state):
        """
        Args:
          - X: torch.Size([BatchSize, NumSteps])
          - state: Tuple[enc_outputs, enc_hidden_state, enc_valid_lens]
              - enc_outputs: torch.Size([BatchSize, NumSteps, num_hiddens])
              - enc_hidden_state:
                - GRU: torch.Size([num_layers, BatchSize, num_hiddens])
                - LSTM: Tuple[torch.Size([num_layers, BatchSize, num_hiddens]), torch.Size([num_layers, BatchSize, num_hiddens])]
              - enc_valid_lens: torch.Size([BatchSize])
        """
        enc_outputs, hidden_state, enc_valid_lens, remove_heads_idx = state
        X = self.embedding(X).permute(1, 0, 2)  # torch.Size([NumSteps, BatchSize, embed_size])
        outputs, self._attention_weights = [], []
        # 每一个step分别进行计算
        # x: torch.Size([BatchSize, embed_size])
        for x in X:
            # query: torch.Size([BatchSize, 1, num_hiddens])
            # context: torch.Size([BatchSize, 1, num_hiddens])
            # core_attention_weights: torch.Size([BatchSize*num_heads, 1, NumSteps])
            if isinstance(hidden_state, tuple):
                query = torch.unsqueeze(hidden_state[0][-1], dim=1)
            else:
                query = torch.unsqueeze(hidden_state[-1], dim=1)
            context, core_attention_weights = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens, remove_heads_idx
            )

            # 合并当前 step 的 x 和 context
            x = torch.cat((context, x.unsqueeze(1)), dim=-1)  # torch.Size([BatchSize, 1, num_hiddens+embed_size])
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(core_attention_weights)
        output = self.dense(torch.cat(outputs, dim=0))
        return output.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens, remove_heads_idx]

    @property
    def attention_weights(self):
        return self._attention_weights


class MultiHeadAttentionDecoder(Seq2SeqAttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, num_heads, dropout=0, **kwargs):
        super().__init__(vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs)
        self.attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, dropout)


class TransformerDecoder(AttentionDecoder):
    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"block{str(i)}",
                DecoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    i,
                ),
            )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state, step: int | None = None):
        """
        Args:
          - X: torch.Size([BatchSize, NumSteps])
          - state: Tuple[enc_outputs, enc_valid_lens, 每一个DecoderBlock用于掩码多头注意力的key_value]
        """
        X = self.pos_encoding(
            self.embedding(X) * math.sqrt(self.num_hiddens), step
        )  # torch.Size([BatchSize, NumSteps, num_hiddens])
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.core_attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.core_attention.attention_weights

        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
