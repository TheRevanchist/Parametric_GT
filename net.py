import torch
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
from utils import prepare_dataset, prepare_loader_train, prepare_loader_val, misc, create_dict_nets_and_features, create_net


def main():
    user = os.path.expanduser("~")
    user = os.path.join(user, 'PycharmProjects/boosting_classifier_with_games')

    current_dataset = 'sun'
    max_epochs = 1
    batch_size = 8

    dataset, stats, number_of_classes = misc(user, current_dataset)
    dataset_train, dataset_val, dataset_test = prepare_dataset(dataset)

    out_dir = os.path.join(os.path.join(os.path.join(user, 'Results'), current_dataset), 'nets')

    nets_and_features = create_dict_nets_and_features()
    net_types = ['resnet18', 'densenet121', 'inception']

    for net_type in net_types:
        if net_type == 'inception':
            inception = 1
        else:
            inception = 0
        train_loader = prepare_loader_train(dataset_train, stats, batch_size, inception=inception)
        val_loader = prepare_loader_val(dataset_val, stats, batch_size, inception=inception)
        test_loader = prepare_loader_val(dataset_test, stats, batch_size, inception=inception)

        net, feature_size = create_net(number_of_classes, nets_and_features, net_type=net_type)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=3e-4)

        best_net = train(net, net_type, train_loader, val_loader, optimizer, criterion, max_epochs, out_dir)
        net.load_state_dict(torch.load(best_net))
        net_accuracy = evaluate(net, test_loader, net_type)
        print('Accuracy: ' + str(net_accuracy))


def train(net, net_type, train_loader, val_loader, optimizer, criterion, epochs, out_dir):
    accuracy = 0
    early_stopping = 0

    for epoch in range(epochs):
        net.train()
        print(net_type + ' --------- ' + 'Epoch: ' + str(epoch))
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
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
        new_accuracy = evaluate(net, val_loader, net_type)
        print(new_accuracy)

        if new_accuracy > accuracy:
            accuracy = new_accuracy

            if epoch > 0:
                os.remove(net_name)
            net_name = out_dir + '/' + net_type + '_epochs' + str(epoch) + '.pth'
            torch.save(net.state_dict(), net_name)
            early_stopping = 0
        else:
            early_stopping += 1
            if early_stopping == 5:  # if there is no improvement on 5 epochs, stop the training
                break

    return net_name


def evaluate(net, test_loader, net_type):
    net.eval()
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(torch.autograd.Variable(inputs))
        if net_type == 'inception':
            outputs = outputs[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = correct / float(total)
    return accuracy


if __name__ == "__main__":
    main()