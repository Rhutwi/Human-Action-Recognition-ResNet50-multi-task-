import torch
import torch.nn as nn
from torchvision.models import resnet50

class MultiTaskResNet50(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        base = resnet50(weights="IMAGENET1K_V2")
        base.fc = nn.Identity()
        self.backbone = base
        self.dropout = nn.Dropout(p=0.2)
        self.head_action = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, num_actions))
        self.head_person = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        f = self.backbone(x)
        f = self.dropout(f)
        action_logits = self.head_action(f)
        person_logit = self.head_person(f).squeeze(1)
        return action_logits, person_logit
