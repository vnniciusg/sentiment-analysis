import torch
import torch.nn as nn

from layers.attention import BahdanauAttention


class LSTMWithAttention(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.5):
        super(LSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = BahdanauAttention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor) -> tuple:

        embedded = self.dropout(self.embedding(text))

        lstm_output, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        context, attn_weights = self.attention(hidden, lstm_output)

        output = self.fc(self.dropout(context)).squeeze(1)

        return torch.sigmoid(output), attn_weights
