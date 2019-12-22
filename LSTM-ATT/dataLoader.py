import torch
import torch.nn as nn
from torchtext.datasets import text_classification
from torchtext.data import Dataset, Field, LabelField, TabularDataset, BucketIterator
from torchtext.vocab import GloVe

import os
import random


root_dir = 'data'
if not os.path.exists(root_dir):
    os.mkdir(root_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEXT = Field(init_token="<BOS>", eos_token="<EOS>",
             lower=True, batch_first=True)
LABEL = LabelField(sequential=False, use_vocab=False, is_target=True)
train_dataset, test_dataset = TabularDataset.splits(
    path="data/ag_news_csv/", train="train.csv", test="test.csv", format="csv",
    fields=[("Label", LABEL), ("Text", TEXT)])
train_dataset, val_dataset = train_dataset.split(random_state=random.seed(33))
TEXT.build_vocab(train_dataset, max_size=25000)

train_iter = BucketIterator(train_dataset, batch_size=128)
val_iter = BucketIterator(val_dataset, batch_size=128)
test_iter = BucketIterator(test_dataset, batch_size=128, train=False)

batch = next(iter(train_iter))


def train(model, train_loader, optimizer,  epochs=20):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for idx, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_pred = model(x_train)
            y_train = y_train - 1
            loss = loss_fn(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_acc = test(model, val_iter)
            if idx % 100 == 0:
                print("Epoch:{}, Train loss:{}, Test Acc:{}".format(epoch, loss.item(),
                                                                    test_acc))


def test(model, data):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    test_acc = 0.
    loss = 0.
    for idx, (x, y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_ = model(x)
        y = y - 1
        loss += loss_fn(y_, y)
        pred = y_.max(-1, keepdim=True)[1]  # .max()分别输出最大值和最小值的index
        test_acc += pred.eq(y.view_as(pred)).sum().item()
    loss /= len(data.dataset)
    print("\n Test Set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        loss, test_acc, len(data.dataset), 100. * test_acc / len(data.dataset)))
    model.train()
    return test_acc / len(data.dataset)

