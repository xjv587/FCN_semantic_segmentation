import torch
import numpy as np

from .models import FCN, save_model
from .utils import DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, load_dense_data, DenseSuperTuxDataset
from torch.utils.data import DataLoader
from . import dense_transforms
import torch.utils.tensorboard as tb
from torch.autograd import Variable

class FocalLoss(torch.nn.Module):
    def __init__(self, weight, gamma=2):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target):
        input = input.view(input.size(0), input.size(1), -1)
        input = input.transpose(1,2)
        input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.weight.type() != input.data.type():
            self.weight = self.weight.type_as(input.data)
        wt = self.weight.gather(0, target.data.view(-1))
        logpt = logpt * Variable(wt)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean()        
    
transform = dense_transforms.Compose([
    dense_transforms.RandomHorizontalFlip(0.3),
    dense_transforms.ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8,1.2), hue=(-0.1,0.1)),
    dense_transforms.ToTensor()
    ])

weight = torch.tensor(DENSE_CLASS_DISTRIBUTION, dtype=torch.float32)
wt = torch.exp(-weight)
wt = wt / wt.sum()
wt = 1000*wt

def train(args):
    from os import path
    model = FCN()
    #train_logger, valid_logger = None, None
    #if args.log_dir is not None:
    #    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
    #    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    train_data = DenseSuperTuxDataset('dense_data/train', transform=transform)
    train_loader = DataLoader(train_data, num_workers=0, batch_size=32, shuffle=True, drop_last=True)
    valid_loader = load_dense_data('dense_data/valid', batch_size=32)
    criterion = torch.nn.CrossEntropyLoss(weight=wt)
    #criterion = FocalLoss(weight=wt)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 20], gamma=0.1)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        confusion_matrix = ConfusionMatrix()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(preds, 1)
            confusion_matrix.add(predicted, labels)

        #if i % args.log_interval == 0:
        #        print(f"Epoch {epoch + 1}/{args.num_epochs}, Batch {i}/{len(train_loader)}, Loss: {loss.item()}")
        avg_train_loss = total_loss / len(train_loader)
        gl_acc = confusion_matrix.global_accuracy
        #train_logger.add_scalar('loss', avg_train_loss, epoch)
        #train_logger.add_scalar('iou', confusion_matrix.iou, epoch)
        #train_logger.flush()

        print(f"Epoch {epoch + 1}/{args.num_epochs}, Average Training Loss: {avg_train_loss}, Global Acc: {gl_acc}, Average Acc: {confusion_matrix.average_accuracy}")

        model.eval()
        total_val_loss = 0.0
        val_confusion_matrix = ConfusionMatrix()

        with torch.no_grad():
            for i, (images, labels) in enumerate (valid_loader):
                images, labels = images, labels
                preds = model(images)
                val_loss = criterion(preds, labels.long())
                total_val_loss += val_loss.item()

                _, predicted = torch.max(preds, 1)
                val_confusion_matrix.add(predicted, labels)

        # Log validation loss and metrics
        avg_val_loss = total_val_loss / len(valid_loader)
        val_iou = val_confusion_matrix.iou
        class_acc = val_confusion_matrix.class_accuracy
        scheduler.step()
        #train_logger.add_scalar('val_loss', avg_val_loss, epoch)
        #train_logger.add_scalar('val_iou', val_iou, epoch)
        #train_logger.flush()

        print(f"Epoch {epoch + 1}/{args.num_epochs}, Average Validation Loss: {avg_val_loss}, Validation IoU: {val_iou}, Validation Class Accuracy: {class_acc}")

    """
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=50)
    #parser.add_argument('--log_interval', type=int, default=10)
    #parser.add_argument('--img_log_interval', type=int, default=100)

    args = parser.parse_args()
    train(args)
