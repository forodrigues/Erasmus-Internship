import torch
import torch.nn as nn
import timm

"""
class MMDModel(nn.Module):
    def __init__(self, backbone_name, in_channels=3, num_classes=1, pretrained=True, drop_rate=0.3, device="cpu"):
        super().__init__()
        self.device = device
        
        # 1. Backbone
        self.model = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,       
            global_pool='avg',   
            drop_rate=drop_rate
        )
        
        feature_dim = self.model.num_features
        
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 3. Classificador (agora recebe 256)
        self.classifier = nn.Linear(256, num_classes)
        
        self.to(device)

    def forward(self, x, return_features=False):

        features_raw = self.model(x)       
        features_neck = self.bottleneck(features_raw)
        
        # 3. Classificar
        logits = self.classifier(features_neck)
        
        if return_features:
            # Retornamos as features do bottleneck para o CMD Loss
            return logits, features_neck
            
        return logits
"""
class MMDModel(nn.Module):
    def __init__(self, backbone_name, in_channels=3, num_classes=1, 
                 pretrained=True, drop_rate=0.3, device="cpu"):
        super().__init__()
        self.device = device
        
        # Create the backbone using timm
        self.model = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=0.3
        )
        
        self.to(device)

    def forward(self, x, return_features=False):

        features_spatial = self.model.forward_features(x)
        logits = self.model.forward_head(features_spatial)
        
        if return_features:
            embedding = self.model.forward_head(features_spatial, pre_logits=True)
            return logits, embedding
            
        return logits

