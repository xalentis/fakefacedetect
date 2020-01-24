import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import torch.utils.data
import numpy as np

BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
EPOCHS = 14
SEED = 1
LOG_INTERVAL = 10
LR = 0.0001
SIZE = 256
PATIENCE = 3

criterion = nn.CrossEntropyLoss()
torch.manual_seed(SEED)
device = torch.device("cuda")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saved.')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


class MesoInception4(nn.Module):

    def __init__(self, num_classes=2):
        super(MesoInception4, self).__init__()
        self.num_classes = num_classes
        # InceptionLayer1
        self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)

        # InceptionLayer2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        # Normal Layer
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    # InceptionLayer
    def InceptionLayer1(self, input):
        x1 = self.Incption1_conv1(input)
        x2 = self.Incption1_conv2_1(input)
        x2 = self.Incption1_conv2_2(x2)
        x3 = self.Incption1_conv3_1(input)
        x3 = self.Incption1_conv3_2(x3)
        x4 = self.Incption1_conv4_1(input)
        x4 = self.Incption1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)

        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        x2 = self.Incption2_conv2_1(input)
        x2 = self.Incption2_conv2_2(x2)
        x3 = self.Incption2_conv3_1(input)
        x3 = self.Incption2_conv3_2(x3)
        x4 = self.Incption2_conv4_1(input)
        x4 = self.Incption2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)

        return y

    def forward(self, input):
        x = self.InceptionLayer1(input)  # (Batch, 11, 128, 128)
        x = self.InceptionLayer2(x)  # (Batch, 12, 64, 64)

        x = self.conv1(x)  # (Batch, 16, 64 ,64)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (Batch, 16, 32, 32)

        x = self.conv2(x)  # (Batch, 16, 32, 32)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x)  # (Batch, 16, 8, 8)

        x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x)  # (Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


train_dataset = datasets.ImageFolder('datasets/train/',
                                     transform=transforms.Compose([transforms.Resize((SIZE, SIZE)),
                                                                   #transforms.RandomHorizontalFlip(),
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize([0.5] * 3, [0.5] * 3)]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, num_workers=0, shuffle=True)

valid_dataset = datasets.ImageFolder('datasets/test/',
                                     transform=transforms.Compose([transforms.Resize((SIZE, SIZE)),
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize([0.5] * 3, [0.5] * 3)]))

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE_TEST, num_workers=0, shuffle=True)

#image_means = torch.stack([t.mean(1).mean(1) for t, c in train_dataset])
#print('Mean: ' + str(image_means.mean(0)))
#image_std = torch.stack([t.std(1).std(1) for t, c in train_dataset])
#print('Std: ' + str(image_std.std(0)))



model = MesoInception4().to(device)
optimizer = Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

train_dataset_size = len(train_dataset)
val_dataset_size = len(valid_dataset)
best_model_wts = model.state_dict()
best_acc = 0.0
iteration = 0
for epoch in range(EPOCHS):
    ###################
    # train the model #
    ###################
    print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
    print('-' * 10)
    model = model.train()
    train_loss = 0.0
    train_corrects = 0.0
    val_loss = 0.0
    val_corrects = 0.0
    for (image, labels) in train_loader:
        image = image.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(image)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        iter_loss = loss.data.item()
        train_loss += iter_loss
        iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
        train_corrects += iter_corrects
        iteration += 1
        if not (iteration % 20):
            print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / BATCH_SIZE_TRAIN,
                                                                       iter_corrects / BATCH_SIZE_TRAIN))
    epoch_loss = train_loss / train_dataset_size
    epoch_acc = train_corrects / train_dataset_size
    print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    ######################
    # validate the model #
    ######################
    model.eval()
    with torch.no_grad():
        for (image, labels) in valid_loader:
            image = image.cuda()
            labels = labels.cuda()
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.data.item()
            val_corrects += torch.sum(preds == labels.data).to(torch.float32)
        epoch_loss = val_loss / val_dataset_size
        epoch_acc = val_corrects / val_dataset_size
        print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    scheduler.step()
    if not (epoch % 10):
        torch.save(model.state_dict(), 'checkpoint_' + str(epoch) + '.pt')

print('Best val Acc: {:.4f}'.format(best_acc))
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'best.pt')

