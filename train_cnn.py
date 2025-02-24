from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
from torchvision import transforms
import torch.optim as optim
import tempfile
import os

log_dir = tempfile.mkdtemp()
train_logger = tb.SummaryWriter(os.path.join(log_dir, 'train'), flush_secs=1)
valid_logger = tb.SummaryWriter(os.path.join(log_dir, 'valid'), flush_secs=1)

tdata_path = "data/train"
vdata_path = "data/valid"

batch_size = 32
learning_rate = 0.02
num_epochs = 100


def train(args):
    model = CNNClassifier()
 
    train_dataset = load_data(tdata_path, num_workers=0, batch_size=batch_size)
    valid_dataset = load_data(vdata_path, num_workers=0, batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 30, 40], gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(train_dataset, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            global_step = epoch * len(train_dataset) + i
            #train_logger.add_scalar('loss', loss.item(), global_step=global_step)

        train_accuracy = 100 * correct_train / total_train
        train_logger.add_scalar('accuracy', train_accuracy, global_step=epoch)

        model.eval()
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for data in valid_dataset:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        valid_accuracy = 100 * correct_valid / total_valid
        valid_logger.add_scalar('accuracy', valid_accuracy, global_step=epoch)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataset):.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Valid Acc: {valid_accuracy:.2f}%')
        
        best_valid_accuracy = 0.0
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 3:
                print("Early stopping!")
                break        

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
