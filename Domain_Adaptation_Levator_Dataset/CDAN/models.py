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

class RandomizedMultiLinearMap(nn.Module):
    def __init__(self, feature_dim, num_classes, output_dim=1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.register_buffer('nw_f', torch.randn(feature_dim, output_dim))
        self.register_buffer('nw_g', torch.randn(num_classes, output_dim))

    def forward(self, f, g):
        proj_f = torch.mm(f, self.nw_f) / (f.size(1) ** 0.5) # Normalização de Xavier implícita
        proj_g = torch.mm(g, self.nw_g) / (g.size(1) ** 0.5)
        return proj_f * proj_g

class DANNModel(nn.Module):
    def __init__(self, backbone_name, in_channels=1, num_classes=1, pretrained=True, drop_rate=0.0, device="cuda"):      
        super().__init__()
        self.device = device

        # 1. Carregar backbone sem o classifier head (num_classes=0) para flexibilidade
        self.model = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0, # Retorna apenas features
            global_pool='', # Vamos fazer pooling manual
            drop_rate=0.3,  # Controlamos dropout manualmente
            drop_path_rate=0.3
        )
        
        feature_dim = self.model.num_features
        
        # 2. Cabeçalho de Classificação Manual (para garantir consistência)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # 3. CDAN Components
        map_dim = 1024
        self.cdan_map = RandomizedMultiLinearMap(feature_dim, num_classes=2, output_dim=map_dim)
        
        self.domain_classifier = nn.Sequential(
            spectral_norm(nn.Linear(map_dim, 1024)), 
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU é melhor para gradientes adversariais
            
            spectral_norm(nn.Linear(1024, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            spectral_norm(nn.Linear(256, 1))        
        )
        self.to(device)

    def forward(self, x, lambda_grl=0.0, return_features=False):

        features_spatial = self.model(x) 
        features_pooled = self.pooling(features_spatial)
        features_flat = self.flatten(features_pooled)
        
       
        features_drop = self.dropout(features_flat)
        task_out = self.classifier(features_drop)

        
        prob_1 = torch.sigmoid(task_out)
        prob_0 = 1.0 - prob_1
        probs_concat = torch.cat((prob_0, prob_1), dim=1) 
        
        features_norm = F.normalize(features_flat, p=2, dim=1)
        h = self.cdan_map(features_norm, probs_concat)
        
        h_grl = GradientReversalLayer.apply(h, lambda_grl)
        domain_out = self.domain_classifier(h_grl)

        if return_features:
            return task_out, domain_out, features_flat
        return task_out, domain_out