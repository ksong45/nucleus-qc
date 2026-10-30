"""Five-stream EfficientNet-B0 classifier (identical to the training code)."""
import torch
import torch.nn as nn
from torchvision import models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
INPUT_SIZE = 224

# Same as the test-time transform used in training.
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class FiveViewNet(nn.Module):
    """One EfficientNet-B0 per view (no weight sharing), concatenated features,
    and a 512-128-2 head. Output: logits for (accept, reject)."""

    def __init__(self, n_views=5):
        super().__init__()

        def stream():
            m = models.efficientnet_b0(weights=None)    # weights come from the checkpoint
            m.classifier = nn.Identity()
            return m

        self.streams = nn.ModuleList([stream() for _ in range(n_views)])
        self.classifier = nn.Sequential(
            nn.Linear(1280 * n_views, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, *views):
        return self.classifier(torch.cat([s(v) for s, v in zip(self.streams, views)], dim=1))


def load_model(path, device):
    net = FiveViewNet()
    net.load_state_dict(torch.load(path, map_location=device))
    return net.to(device).eval()
