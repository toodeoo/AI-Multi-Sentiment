import torch
from torch import nn
from model.baseline import XLM_R, Convnext

class MultiModel(nn.Module):
    def __init__(self, args):
        super(MultiModel, self).__init__()
        self.TextModel = XLM_R(args)
        self.ImgModel = Convnext(args)
        self.strategy = args.fuse_strategy

        self.linear_text_k1 = nn.Linear(1000, 1000)
        self.linear_text_v1 = nn.Linear(1000, 1000)
        self.linear_img_k2 = nn.Linear(1000, 1000)
        self.linear_img_v2 = nn.Linear(1000, 1000)
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=1000, num_heads=2, batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.classifier_multi = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(2000, 1000),
            # nn.ReLU(),
            # nn.Dropout(p=args.dropout),
            # nn.Linear(1000, 200),
            # nn.ReLU(),
            # nn.Linear(200, 40),
            # nn.ReLU(),
            # nn.Linear(40, 3),
            nn.Linear(2000, 3)
        )
        self.classifier_only = nn.Linear(1000, 3)
        # self.classifier_only = nn.Sequential(
            # nn.Linear(1000, 500),
            # nn.ReLU(),
            # nn.Linear(500, 200),
            # nn.ReLU(),
            # nn.Linear(200, 40),
            # nn.ReLU(),
            # nn.Linear(40, 3),
        # )
        
    def forward(self, batch_text=None, batch_img=None):
        if batch_text is None and batch_img is None:
            return None, None, None
        elif batch_text is not None and batch_img is None:
            _, text = self.TextModel(batch_text)
            text_out = self.classifier_only(text)
            return text_out, None, None
        elif batch_text is None and batch_img is not None:
            img = self.ImgModel(batch_img)
            img_out = self.classifier_only(img)
            return None, img_out, None
        else:
            _, text_out = self.TextModel(batch_text)
            img_out = self.ImgModel(batch_img)

        if self.strategy == 'cat':
            multi_out = torch.cat((text_out, img_out), 1)
        elif self.strategy == 'attention':
            multi_out = self.attention(text_out, img_out)
        else:
            multi = torch.stack((text_out, img_out), dim=1)
            multi_out = self.transformer_encoder(multi)

        text_out = self.classifier_only(text_out)
        img_out = self.classifier_only(img_out)
        multi_out = self.classifier_multi(multi_out)
        return text_out, img_out, multi_out

    def attention(self, text_out, img_out):
        text_k1 = self.linear_text_k1(text_out)
        text_v1 = self.linear_text_v1(text_out)
        img_k2 = self.linear_img_k2(img_out)
        img_v2 = self.linear_img_v2(img_out)

        k = torch.stack((text_k1, img_k2), dim=1)
        v = torch.stack((text_v1, img_v2), dim=1)
        query = torch.stack((text_out, img_out), dim=1)
        out, w = self.multi_head_attn(query, k, v)
        return out