import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.nn.utils import spectral_norm

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class DANNModel(nn.Module):
    def __init__(self, backbone_name, in_channels=1, num_classes=1, pretrained=True, drop_rate=0.0, device="cuda"):      
        super().__init__()
        self.device = device

        # 1. Carregar backbone
        self.model = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0, 
            global_pool='', 
            drop_rate=0.3,  
            drop_path_rate=0.3
        )
        
        feature_dim = self.model.num_features
        
        # 2. Cabeçalho de Classificação
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # 3. Domain Classifier (DANN PURO - SIMPLIFICADO)
        # Removemos o CDAN Map. O input agora é direto da backbone (feature_dim).
        # Adicionamos LayerNorm e Dropout 0.5 para regularizar fortemente.
        self.domain_classifier = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, 256)), 
            nn.LayerNorm(256),              # Estabiliza o treino adversarial
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),                # Dropout alto para dificultar a vida ao discriminador
            
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            spectral_norm(nn.Linear(128, 1))        
        )
        self.to(device)

    def forward(self, x, lambda_grl=0.0, return_features=False):
        # --- Feature Extraction ---
        features_spatial = self.model(x) 
        features_pooled = self.pooling(features_spatial)
        features_flat = self.flatten(features_pooled)
        
        # --- Task Classification ---
        features_drop = self.dropout(features_flat)
        task_out = self.classifier(features_drop)
        
        # --- Domain Classification (Pure DANN) ---
        # Passamos as features diretamente, sem multiplicar por probs (CDAN)
        h_grl = GradientReversalLayer.apply(features_flat, lambda_grl)
        domain_out = self.domain_classifier(h_grl)

        if return_features:
            return task_out, domain_out, features_flat
        return task_out, domain_out