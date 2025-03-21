import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    implementation of Bahdanau (Additive Attention) mechanism for encoder-decoder models.

    Theory:
        Bahdanau Attetion computes a dynamic context vector that allows the decoder to focus on different parts
        of the input sequence at each generation step. The mechanism is defined by three main components:
            1. projection of encoder states (Wa)
            2. projection of decoder states (Ua)
            3. score layer (V) that calculates alignment weights

    Math:
        Given:
            - h_j (encoder_states): the ocult states  (j = 1...T)
            - s_{t-1} (decoder_states): previous decoder hidden state

            Steps:
                1. Attention score calculation:
                    e_{tj} = v^T * tanh(Wa * h_j + Ua * s_{t-1} + b_a)
                2. Softmax normalization:
                    α_{tj} = softmax(e_{tj}) = exp(e_{tj}) /  Σ_{k=1}^T exp(e_{tk})
                3. Context vector:
                    c_t =  Σ_{j=1}^T α_{tj} * h_j

            Where:
                - Wa ∈ ℝ^ {hidden_size x hidden_size}
                - Ua ∈ ℝ^ {hidden_size x hidden_size}
                - v ∈ ℝ^{hidden_size x 1}
                - b_a is bais (optional)

    Reference:
        Bahdanau et al. (2015) - "Neural Machine Translation by Jointly Learning to Align and Translate" ->  https://arxiv.org/abs/1409.0473
    """

    def __init__(self, hidden_size: int) -> None:
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, decoder_state: torch.Tensor, encoder_states: torch.Tensor) -> tuple:

        # project decoder state: s_{t-1} -> Ua(s_{t-1}) => (batch_size, 1, hidden_size)
        decoder_proj = self.Ua(decoder_state).unsqueeze(1)

        # project encoder states: h_j -> Wa(h_j) => (batch_size, seq_len, hidden_size)
        encoder_proj = self.Wa(encoder_states)

        # calculate attention scores: e_{tj} = v^T * tanh(Wa * h_j + Ua * s_{t-1} + b_a) => (batch_size, seq_len)
        scores = self.V(torch.tanh(decoder_proj + encoder_proj)).squeeze(-1)

        # softmax normalization to get attentions weights => (batch_size, seq_len)
        attn_weights = F.softmax(scores, dim=1)

        # calculate context vector: c_t =  Σ_{j=1}^T α_{tj} * h_j => (batch_size, hidden_size)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_states).squeeze(1)

        return context, attn_weights
