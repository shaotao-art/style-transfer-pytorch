import torch
from torch import nn
from typing import List


# NOTE: this parma is important, it will affect the style transfer result
VGG_SLICE_LST = [2, 6, 11, 20]
# relu1_1 is the first conv from channel 3 to 64
# relu2_1 is the conv right after first maxpooling
# relu3_1 is the conv right after second maxpooling
# relu4_1 is the conv right after third maxpooling
# relu5_1 ...


def get_content_features(img, vgg19, use_relu=False):
    # NOTE: whether should include relu or not?
    if use_relu:
        vgg_relu41 = nn.Sequential(*list(vgg19.features.children())[:VGG_SLICE_LST[-1] + 1])
    else:
        vgg_relu41 = nn.Sequential(*list(vgg19.features.children())[:VGG_SLICE_LST[-1]])
    return vgg_relu41(img)


def get_style_features(img, vgg19, use_relu=False):
    # get style features, output of vgg19's relu1_1, relu2_1, relu3_1, relu4_1
    if use_relu:
        add = 1
    else:
        add = 0
    vgg_relu1_1 = vgg19.features[:VGG_SLICE_LST[0]+add]
    vgg_relu2_1 = vgg19.features[VGG_SLICE_LST[0]+add:VGG_SLICE_LST[1]+add]
    vgg_relu3_1 = vgg19.features[VGG_SLICE_LST[1]+add:VGG_SLICE_LST[2]+add]
    vgg_relu4_1 = vgg19.features[VGG_SLICE_LST[2]+add:VGG_SLICE_LST[3]+add]
    features = []

    x = img
    x = vgg_relu1_1(x)
    features.append(x)
    x = vgg_relu2_1(x)
    features.append(x)
    x = vgg_relu3_1(x)
    features.append(x)
    x = vgg_relu4_1(x)
    features.append(x)
    return features

# use ADIN to get the target features
def adain(content_features, style_features, eps=1e-5):
    assert content_features.size() == style_features.size() and content_features.ndim == 4
    content_mean, content_std = content_features.mean(dim=[2, 3]), content_features.std(dim=[2, 3])
    style_mean, style_std = style_features.mean(dim=[2, 3]), style_features.std(dim=[2, 3])
    # normalize content features
    normalized_content_features = (content_features - content_mean[:, :, None, None]) / torch.clamp(content_std[:, :, None, None], min=eps)
    remaped_content_features = normalized_content_features * style_std[:, :, None, None] + style_mean[:, :, None, None]
    return remaped_content_features

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.pad = nn.ReflectionPad2d((padding, padding, padding, padding))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(self.pad(x)))

def UpsampleBlock():
    return nn.Sequential(
        nn.Upsample(mode='nearest', scale_factor=2)
    )
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            ConvBlock(512, 512, 3, 1, 1),
            UpsampleBlock(),
            ConvBlock(512, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
            UpsampleBlock(),
            ConvBlock(256, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            UpsampleBlock(),
            ConvBlock(128, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, 1, 1, 0)
        )
    def forward(self, x):
        return self.decoder(x)
    



def content_loss_fn(content_features, target_features):
    return ((content_features - target_features) ** 2).mean()

def style_loss_fn(style_features_lst: List[torch.Tensor], target_features_lst: list[torch.Tensor]):
    loss = 0.0
    for style_features, target_features in zip(style_features_lst, target_features_lst):
        style_features_mean = style_features.mean(dim=[2, 3])
        target_features_mean = target_features.mean(dim=[2, 3])
        style_features_std = style_features.std(dim=[2, 3])
        target_features_std = target_features.std(dim=[2, 3])
        single_layer_loss = ((style_features_mean - target_features_mean) ** 2).mean() + ((style_features_std - target_features_std) ** 2).mean()
        loss = loss + single_layer_loss
    return loss


class StyleTransferModel(nn.Module):
    def __init__(self, use_relu, style_loss_w):
        super(StyleTransferModel, self).__init__()
        # get pretrained vgg19 model
        vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.vgg = vgg
        
        self.decoder = Decoder()
        
        self.use_relu = use_relu
        self.style_loss_w = style_loss_w
        
    def train_loss(self, content_img, style_img):
        with torch.no_grad():
            content_features = get_content_features(content_img, 
                                                    self.vgg, 
                                                    use_relu=self.use_relu)
            style_style_features_lst = get_style_features(style_img, 
                                                          self.vgg, 
                                                          use_relu=self.use_relu)
            # the last element of style_style_features_lst is the content feature
            remaped_content_features = adain(content_features, style_style_features_lst[-1])   
            
     
        reconstructed_img = self.decoder(remaped_content_features)
        # NOTE: should not clamp image to (-1.0, 1.0) when use imagenet mean and std
        # NOTE: because after imagenet normalize, value may out of (-1.0, 1.0)
        # NOTE: so we should not use the tanh function in decoder output either
        reconstructed_img_style_feature_lst = get_style_features(reconstructed_img, 
                                                                 self.vgg, 
                                                                 use_relu=self.use_relu)
        reconstructed_img_content_feature = reconstructed_img_style_feature_lst[-1]


        content_loss = content_loss_fn(reconstructed_img_content_feature, remaped_content_features)
        style_loss = style_loss_fn(reconstructed_img_style_feature_lst, style_style_features_lst)
        loss = content_loss + style_loss * self.style_loss_w
        return dict(loss=loss, content_loss=content_loss, style_loss=style_loss)
    
    
    
    def forward(self, content_img, style_img):
        content_features = get_content_features(content_img, 
                                                self.vgg, 
                                                use_relu=self.use_relu)
        style_style_features_lst = get_style_features(style_img, 
                                                        self.vgg, 
                                                        use_relu=self.use_relu)
        # the last element of style_style_features_lst is the content feature
        remaped_content_features = adain(content_features, style_style_features_lst[-1])   
        reconstructed_img = self.decoder(remaped_content_features)
        return reconstructed_img
    
    @torch.no_grad()
    def transfer(self, content_img, style_img):
        return self.forward(content_img, style_img)
        