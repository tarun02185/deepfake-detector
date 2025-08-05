# model.py
import torch
import torch.nn as nn
import timm

class DeepfakeXception(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'xception',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 299, 299)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
