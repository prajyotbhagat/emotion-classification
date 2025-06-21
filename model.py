import torch
import torch.nn as nn
import torchvision.models as models

class VGG16LSTM(nn.Module):
    def __init__(self, num_classes, fine_tune=True):
        super().__init__()


        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3)

        self.feature_extractor = vgg.features


        if not fine_tune:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.lstm_input_size = 512 * 4


        self.lstm1 = nn.LSTM(input_size=self.lstm_input_size, hidden_size=160, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(320)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(input_size=320, hidden_size=160, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(320)
        self.dropout2 = nn.Dropout(0.3)

        self.final_dropout = nn.Dropout(0.1)


        self.fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute(0, 3, 1, 2)
        B, T, C, H = x.size()
        x = x.reshape(B, T, C * H)

        x, _ = self.lstm1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout2(x)

        x = torch.max(x, dim=1).values
        x = self.final_dropout(x)
        return self.fc(x)

  
