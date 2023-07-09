from torch import nn
from transformers import XLMRobertaModel
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights

class XLM_R(nn.Module):
    def __init__(self, args):
        super(XLM_R, self).__init__()
        self.encoder = XLMRobertaModel.from_pretrained(args.pretrained)
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.transform = nn.Sequential(
            nn.Linear(768, 1000),
            nn.ReLU(),
        )

    def forward(self, x):
        feature = self.encoder(**x)
        h = feature.last_hidden_state
        out = self.transform(feature.pooler_output)
        return h, out


class Convnext(nn.Module):
    def __init__(self, args):
        super(Convnext, self).__init__()
        self.encoder = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.encoder(x)
        return out