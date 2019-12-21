import os

from easy_mnist.mnist_data import mnist_data
from easy_mnist.models import *


def train(model, train_loader, loss_fn, optimizer, epochs=10):
    model.train()
    best_accuracy = 0.
    for epoch in range(1, epochs+1):
        for idx, (x_train, y_train) in enumerate(train_loader):
            # 前向传播
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 50 == 0:
                print("Train Epoch:{} [{}/{} ({})%]\t Total Loss: {}".format(
                   epoch,  idx * len(x_train), len(train_loader.dataset),
                    100. * (idx / len(train_loader)), loss.item()
                ))
        acc = test(model, test_loader, loss_fn)
        if best_accuracy < acc:
            best_accuracy = acc
            model_dir = 'model'
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            torch.save(model.state_dict(), os.path.join(model_dir, "cnn_model.ckpt"))
    print("Best Accuracy:{}".format(best_accuracy))


def test(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in test_loader:
            outputs = model(images)
            test_loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum()
    test_loss /= len(test_loader)
    model.train()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


if __name__ == '__main__':
    input_size = 28 * 28
    hidden_size = 128
    output_size = 10
    train_loader, test_loader = mnist_data()
    loss_fn = nn.CrossEntropyLoss()
    model = CNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(model, train_loader, loss_fn, optimizer, epochs=10)
