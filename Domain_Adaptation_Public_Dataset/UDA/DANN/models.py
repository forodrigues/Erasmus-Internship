import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Inverte o sinal do gradiente para o treino adversarial
        return grad_output.neg() * ctx.lambda_, None

# -----------------------------
# DANN Model Otimizável
# -----------------------------
class DANNModel(nn.Module):
    def __init__(self, backbone_name, in_channels=3, num_classes=1, pretrained=True, drop_rate=0.3, device="cpu"):      
        super().__init__()
        self.device = device

        # Criamos o modelo sem a cabeça final (num_classes=0) para extrair apenas as features
        self.model = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0, 
            global_pool='avg', # Garante que a saída é um vetor (batch, feature_dim)
            drop_rate=drop_rate,
            drop_path_rate=0.3
        )

        feature_dim = self.model.num_features
        
        # Explicitamos o classificador para podermos dar-lhe um LR diferente no optimizador
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Discriminador de Domínio
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256), # Adicionado para estabilidade no treino adversarial
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.to(device)

    def forward(self, x, lambda_grl=0.0, return_features=False):
        # 1. Extração de Features (Backbone)
        features = self.model(x) 
        
        # 2. Classificação da Tarefa Principal (Source)
        task_out = self.classifier(features)

        # 3. Classificação de Domínio com Inversão de Gradiente (GRL)
        # Passamos as features pela GRL antes do discriminador
        grl_features = GradientReversalLayer.apply(features, lambda_grl)
        domain_out = self.domain_classifier(grl_features)

        if return_features:
            return task_out, domain_out, features
        
        return task_out, domain_out