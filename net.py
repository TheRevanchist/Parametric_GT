import torch
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
from utils import prepare_dataset, prepare_train_loader, prepare_val_loader, misc, create_dict_nets_and_features, create_net


def main():
    user = os.path.expanduser("~")
    user = os.path.join(user, 'PycharmProjects/Parametric_GT')

    current_dataset = 'caltech'
    max_epochs = 20
    batch_size = 8

    dataset, stats, number_of_classes = misc(user, current_dataset)
    dataset_train = os.path.join(dataset, 'train_labelled0.1')
    dataset_test = os.path.join(dataset, 'test')

    nets_and_features = create_dict_nets_and_features()
    net_types = ['resnet18']
    out_dir = os.path.join(os.path.join(os.path.join(user, 'Results'), current_dataset), 'nets')

    for net_type in net_types:
        inception = net_type == 'inception'
        train_loader = prepare_train_loader(dataset_train, stats, batch_size, inception)
        test_loader = prepare_val_loader(dataset_test, stats, batch_size, inception)

        net, feature_size = create_net(number_of_classes, nets_and_features, net_type=net_type)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        best_net = train(net, net_type, train_loader, test_loader, optimizer, criterion, max_epochs, out_dir)

        net.load_state_dict(torch.load(best_net))
        net_accuracy = evaluate(net, test_loader)
        print('Accuracy: ' + str(net_accuracy))


def train(net, net_type, train_loader, val_loader, optimizer, criterion, epochs, out_dir, ind):

    for epoch in range(epochs):
        net.train()
        print(net_type + ' --------- ' + 'Epoch: ' + str(epoch))
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels, index = data
            inputs, labels = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            if net_type == 'inception':
                outputs = outputs[0] + outputs[1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i == 10:  # print stats after 10 mini batches for each epoch
                print('[%d, %5d] loss: %.16f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

        print("Validating results in: " + str(epoch) + "-th epoch")
        new_accuracy = evaluate(net, val_loader)
        print(new_accuracy)

        net_name = out_dir + '/' + net_type + ind + '.pth'
        torch.save(net.state_dict(), net_name)

    return net_name


def evaluate(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels, _ = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(torch.autograd.Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = correct / float(total)
    return accuracy


if __name__ == "__main__":
    main()