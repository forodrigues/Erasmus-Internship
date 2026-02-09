import torch
import torch.nn as nn
import timm


class BaseModel(nn.Module):
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

