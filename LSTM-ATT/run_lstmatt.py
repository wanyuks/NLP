import torch
import torch.nn as nn

from lstm_att.LSTMATT import LSTMATT
from lstm_att.dataLoader import *


VOCAB_SIZE = 25004
EMBED_SIZE = 128
HIDDEN_SIZE = 128
EPOCHS = 10


def main():
    model = LSTMATT(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE, output_size=4,
                    num_layers=2, drop_out=0.2)
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model=model, train_loader=train_iter,
          optimizer=optimizer, epochs=EPOCHS)


if __name__ == '__main__':
    main()
