import torch
import torch.nn as nn

class TabTransformer(nn.Module):

    def __init__(self, input_dim, num_classes=3,
                 d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.num_token_embed = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_dim + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier  = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.num_token_embed.weight)
        nn.init.constant_(self.num_token_embed.bias, 0)

    def forward(self, x):
        b = x.size(0)
        x = self.num_token_embed(x.unsqueeze(-1))
        cls_token = self.cls_token.expand(b, -1, -1)

        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]

        x = self.transformer(x)
        return self.classifier(x[:, 0])

    def extract_latent_features(self, x):
        b = x.size(0)
        x = self.num_token_embed(x.unsqueeze(-1))
        cls_token = self.cls_token.expand(b, 1, -1)
        x = torch.cat([cls_token, x], 1) + self.pos_embedding[:, :x.size(1)]
        return self.transformer(x)[:, 0]
