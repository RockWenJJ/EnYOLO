import torch
import torch.nn as nn

class CoralLoss(nn.Module):
    def __init__(self):
        """
        Initializes the CoralLoss module.
        """
        super(CoralLoss, self).__init__()

    def forward(self, source_features, target_features):
        """
        Forward pass of the CoralLoss module to compute the loss between source and target features.
        
        Args:
        source_features (torch.Tensor): A 2D tensor of shape (batch_size, feature_dim) representing source domain features.
        target_features (torch.Tensor): A 2D tensor of shape (batch_size, feature_dim) representing target domain features.
        
        Returns:
        torch.Tensor: Scalar tensor representing the CORAL loss.
        """
        return self.coral_loss(source_features, target_features)

    def compute_covariance(self, features):
        """
        Computes the covariance matrix of the given features.
        
        Args:
        features (torch.Tensor): A 2D tensor of shape (batch_size, feature_dim).
        
        Returns:
        torch.Tensor: Covariance matrix of shape (feature_dim, feature_dim).
        """
        n = features.size(0)
        features = features.reshape(n, -1)
        features_mean = torch.mean(features, dim=0)
        features = features - features_mean.expand_as(features)
        covariance = 1 / (n - 1) * features.t().mm(features)
        return covariance

    def coral_loss(self, source_features, target_features):
        """
        Computes the CORAL loss between the source and target features.
        
        Args:
        source_features (torch.Tensor): A 2D tensor of shape (batch_size, feature_dim).
        target_features (torch.Tensor): A 2D tensor of shape (batch_size, feature_dim).
        
        Returns:
        torch.Tensor: Scalar tensor representing the CORAL loss.
        """
        source_covariance = self.compute_covariance(source_features)
        target_covariance = self.compute_covariance(target_features)
        loss = torch.norm(source_covariance - target_covariance, p='fro') ** 2
        return loss
