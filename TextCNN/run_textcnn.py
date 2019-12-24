import torch
from TextCNN.model import TextCNN

from dataLoader import device, train, train_iter

VOCAB_SIZE = 25004
EMBED_SIZE = 128
HIDDEN_SIZE = 128
EPOCHS = 10


def main():
    model = TextCNN(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, out_channels=100,
                    kernel_size=[2, 3, 4], dropout_rate=0.2)
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model=model, train_loader=train_iter,
          optimizer=optimizer, epochs=EPOCHS)


if __name__ == '__main__':
    main()
