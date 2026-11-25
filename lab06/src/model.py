import torch.nn as nn


class Lab3BestCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        num_classes: int = 10,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(conv2_channels * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(cfg: dict) -> nn.Module:
    m_cfg = cfg["model"]
    return Lab3BestCNN(
        in_channels=m_cfg["in_channels"],
        conv1_channels=m_cfg["conv1_channels"],
        conv2_channels=m_cfg["conv2_channels"],
        num_classes=m_cfg["num_classes"],
    )