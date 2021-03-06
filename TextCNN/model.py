import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, out_channels,
                 kernel_size=[2, 3, 4], dropout_rate=0.2, num_class=4):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.embed_size,
                                    out_channels=self.out_channels,
                                    kernel_size=h),
                          nn.ReLU(),)
            for h in self.kernel_size
        ])
        self.fc = nn.Linear(self.out_channels * len(self.kernel_size), self.num_class)

    def forward(self, x):
        #: (batch_size, seq_len)

        x = self.embed(x)
        #: (batch_size, seq_len, embed_size)

        x = x.permute(0, 2, 1)   # 因为以为卷积是在最后一个维度上扫描的，所以得把seq_len调到最后一个维度上
        # x: (batch_size, embed_size, seq_len)

        outs = [conv1d(x) for conv1d in self.convs]

        outs = [F.max_pool1d(out, out.shape[2]).squeeze(2)
                for out in outs]

        out = torch.cat(outs, dim=1)
        # print(out.shape)
        out = F.dropout(out, p=self.dropout_rate)

        out = self.fc(out)
        return out
