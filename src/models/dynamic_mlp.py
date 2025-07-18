import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(hidden_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.activation = nn.Mish()

    def forward(self, x):
        return self.activation(self.block(x) + x)

class DynamicFeatureMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Mish())
        self.res_block = ResBlock(128, 64)
        self.output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish())
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        feats = self.output_layer(self.res_block(self.input_layer(x)))
        return self.classifier(feats)

    def extract_latent_features(self, x):
        return self.output_layer(self.res_block(self.input_layer(x)))
