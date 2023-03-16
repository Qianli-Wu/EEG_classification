import torch
import torch.nn as nn

class torch_hybrid_cnn_lstm_model(nn.Module):
    def __init__(self):
        super(torch_hybrid_cnn_lstm_model, self).__init__()

        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(22, 25, (10, 1), padding=(6, 0)),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            nn.BatchNorm2d(25),
            nn.Dropout(0.5)
        )

        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, (10, 1), padding=(5, 0)),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            nn.BatchNorm2d(50),
            nn.Dropout(0.5)
        )

        # Conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, (10, 1), padding=(6, 0)),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            nn.BatchNorm2d(100),
            nn.Dropout(0.5)
        )

        # Conv block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, (10, 1), padding=(6, 0)),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            nn.BatchNorm2d(200),
            nn.Dropout(0.5)
        )

        # FC + LSTM layers
        self.fc1 = nn.Linear(800, 4)
        # self.lstm = nn.LSTM(100, 10, batch_first=True, dropout=0.6)

        # Output layer
        # self.fc2 = nn.Linear(10, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        # print(f"Output shape after fc1: {x.shape}")
        x = self.conv2(x)
        # print(f"Output shape after fc2: {x.shape}")
        x = self.conv3(x)
        # print(f"Output shape after fc3: {x.shape}")
        x = self.conv4(x)
        # print(f"Output shape after fc4: {x.shape}")

        x = torch.flatten(x, 1)
        # print(f"Output shape after Flatten: {x.shape}")
        x = self.fc1(x)
        # print(f"Output shape after fc1: {x.shape}")
        # x = x.unsqueeze(2)
        # print(f"Output shape after unsqueeze: {x.shape}")
        # x, _ = self.lstm(x)
        # x = x[:, -1, :]
        
        # x = self.fc2(x)
        x = self.softmax(x)
        return x

# Create the model
hybrid_cnn_lstm_model = torch_hybrid_cnn_lstm_model()

# Print the model summary
print(hybrid_cnn_lstm_model)
