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
        return grad_output.neg() * ctx.lambda_, None

class RandomizedMultiLinearMap(nn.Module):
    def __init__(self, feature_dim, num_classes, output_dim=1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.nw_f = nn.Linear(feature_dim, output_dim, bias=False)
        self.nw_g = nn.Linear(num_classes, output_dim, bias=False)
     
        with torch.no_grad():
            nn.init.normal_(self.nw_f.weight, 0, 1)
            nn.init.normal_(self.nw_g.weight, 0, 1)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, f, g):
        return self.nw_f(f) * self.nw_g(g)

class DANNModel(nn.Module):
    def __init__(self, backbone_name, in_channels=3, num_classes=1, pretrained=True, drop_rate=0.3, device="cpu"):      
        super().__init__()
        self.device = device

        self.model = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0, 
            global_pool='avg',
            drop_rate=drop_rate,
            drop_path_rate=0.3
        )

        feature_dim = self.model.num_features
        map_dim = 512
        
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        self.cdan_map = RandomizedMultiLinearMap(feature_dim, num_classes=2, output_dim=map_dim)

        self.domain_classifier = nn.Sequential(
            nn.Linear(map_dim, 1024),  
            nn.BatchNorm1d(1024),      
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),          
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),           
            
            nn.Linear(1024, 1)        
        )

        self.to(device)

    def forward(self, x, lambda_grl=0.0, return_features=False):
        

        features = self.model(x) 
        
        task_out = self.classifier(features)

        

        prob_1 = torch.sigmoid(task_out)
        prob_0 = 1.0 - prob_1
        probs_concat = torch.cat((prob_0, prob_1), dim=1) 

        # 4. Cálculo do Domínio
        h = self.cdan_map(features, probs_concat)
        h_grl = GradientReversalLayer.apply(h, lambda_grl)
        domain_out = self.domain_classifier(h_grl)

        if return_features:
            return task_out, domain_out, features

        return task_out, domain_out