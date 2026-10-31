import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from params import OUT_DIM, HIDDEN_DIMS, NEGATIVE_SLOPE, DROPOUT_P, TRIPLET_MARGIN

class SiameseNet(nn.Module):
    """
    Siamese Network with Resnet18 sub-network.
    """
    
    def __init__(self, embed_dim=OUT_DIM):
        super().__init__()

        try:
            subnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            subnet = models.resnet18(pretrained=True)

        subnet.fc = nn.Identity()
        self.subnet = subnet

        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
        Returns:
            torch.Tensor: L2-normalized embeddings of shape (batch_size, embed_dim)
        """
        # Extract features
        features = self.subnet(x)

        # Project to embedding space
        embeddings = self.head(features)

        # L2 normalization for stable metric learning
        return F.normalize(embeddings, dim=1)

class TripletLoss(nn.Module):

    def __init__(self, margin=TRIPLET_MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        """
        Compute squared Euclidean distance per sample
        Args:
            x1 (torch.Tensor): tensor of shape (B, D)
            x2 (torch.Tensor): tensor of shape (B, D)
        Returns:
            torch.Tensor: distances of shape (B,) with squared L2 values
        """
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Calculate triplet margin loss for three input embeddings
        Args: anchor, positive, negative embeddings (B, D)
        Returns: mean loss over batch as a scalar tensor
        """
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()