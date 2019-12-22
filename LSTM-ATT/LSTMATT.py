import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMATT(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size,
                 num_layers=5, drop_out=0.5):
        super(LSTMATT, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        self.output_size = output_size
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size,
                            num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        self.dp = nn.Dropout(self.drop_out)
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)

        self.W_w = nn.Parameter(torch.Tensor(self.hidden_size*2, self.hidden_size*2))
        self.u_w = nn.Parameter(torch.Tensor(self.hidden_size*2, 1))

        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

    def forward(self, x):
        # x: [batch_size, seq_len]

        x = self.dp(self.embed(x))
        # x: [batch_size, seq_len, embed_size]

        outputs, hidden = self.lstm(x)
        # hidden: [num_layers * directions, batch_size, hidden_size]
        # outputs: [batch_size, seq_len, hidden_size * 2]

        # outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, seq_len, hidden_size * 2]

        # 实现tanh attention
        score = torch.tanh(torch.matmul(outputs, self.W_w))
        # score: [batch_size, real_seq, hidden_size * 2]

        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        # attention_weights: [batch_size, real_seq, 1]

        scored_x = outputs * attention_weights
        # scored_x : [batch_size, real_seq, hidden_size * 2]

        feat = torch.sum(scored_x, dim=1)
        # feat : [batch_size, hidden_size * 2]

        return self.fc(feat)


