from torch import nn
from torchvision.models.vgg import vgg19


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        feature_extractor = nn.Sequential(*list(vgg19_model.features)[:31]).eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = feature_extractor
        self.mse = nn.MSELoss()

    def forward(self, sr, hr):
        sr = self.feature_extractor(sr)
        hr = self.feature_extractor(hr)
        perceptual_loss = self.mse(sr, hr)
        return perceptual_loss