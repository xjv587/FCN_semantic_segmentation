import torch
import torch.nn.functional as F

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5, stride=3, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(32, eps=1e-05, affine=True)
        self.conv2 = torch.nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(96, eps=1e-05, affine=True)
        self.conv3 = torch.nn.Conv2d(96, 512, kernel_size=3, stride=1, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(512, eps=1e-05, affine=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(512, 6)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) #3*64*64-->32*22*22
        x = F.max_pool2d(x, 2, 2) #32*22*22-->32*11*11
        x = F.relu(self.bn2(self.conv2(x))) #32*11*11-->96*6*6
        x = F.max_pool2d(x, 2, 2) #96*6*6-->96*3*3
        x = F.relu(self.bn3(self.conv3(x))) #96*3*3-->512*1*1
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x

class downBlock(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(n_input, n_output, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(n_output, eps=1e-05, affine=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_output, n_output, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(n_output, eps=1e-05, affine=True),
            torch.nn.ReLU(inplace=True))
        
        self.downsample = None
        if n_input != n_output:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(n_output))
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.net(x) + identity
    
class upBlock(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=2, stride=2)
        self.conv = downBlock(n_input, n_output)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class FCN(torch.nn.Module):
    def __init__(self, n_input_channels=3, n_classes=5):
        super().__init__()

        self.intro = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.down1 = downBlock(32, 64)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = downBlock(64, 128)

        self.up1 = upBlock(128, 64)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.up2 = upBlock(64, 32)
        self.bn2 = torch.nn.BatchNorm2d(32)
        
        self.fc1 = torch.nn.Conv2d(32, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.intro(x)
        x_1 = self.intro(x)
        if min(x_1.size(2), x_1.size(3)) >= 2:
            x_1 = self.maxpool(x_1)        
        x2 = self.down1(x_1)
        x_2 = self.down1(x_1)
        if min(x_2.size(2), x_2.size(3)) >= 2:
            x_2 = self.maxpool(x_2)
        x3 = self.down2(x_2)
      
        x4 = self.up1(x3, x2)
        x4 = self.bn1(x4)
        x5 = self.up2(x2, x1)
        x5 = self.bn2(x5)
        x = F.relu(self.fc1(x5))
        return x


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
