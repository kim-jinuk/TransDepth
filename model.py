import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from batchnorm import SynchronizedBatchNorm2d

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=1280, decoder_width = .6):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)
        
        self.up0 = UpSample(skip_input=features//1 + 320, output_features=features//2)
        self.up1 = UpSample(skip_input=features//2 + 160, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 64, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 32, output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 +  24, output_features=features//8)
        self.up5 = UpSample(skip_input=features//8 +  16, output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4,x_block5,x_block6 = features[2], features[4], features[6], features[9], features[15],features[18],features[19]
        x_d0 = self.conv2(x_block6)
        x_d1 = self.up0(x_d0, x_block5)
        x_d2 = self.up1(x_d1, x_block4)
        x_d3 = self.up2(x_d2, x_block3)
        x_d4 = self.up3(x_d3, x_block2)
        x_d5 = self.up4(x_d4, x_block1)
        x_d6 = self.up5(x_d5, x_block0)
        return self.conv3(x_d6)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.original_model = models.mobilenet_v2( pretrained=True )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

    def get_bn_before_relu(self):
        bn1 = self.original_model.features[3].conv[1][1]
        bn2 = self.original_model.features[6].conv[1][1]
        bn3 = self.original_model.features[13].conv[1][1]
        bn4 = self.original_model.features[18][1]

        return [bn1, bn2, bn3, bn4]

    def get_channel_num(self):
        return [144, 192, 576, 1280]

    def extract_feature(self, x):
        features = [x]
        skip_feat = self.forward(x)
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        t0 = self.original_model.features[3].conv[1][1](self.original_model.features[3].conv[1][0](self.original_model.features[3].conv[0](features[3])))
        t1 = self.original_model.features[6].conv[1][1](self.original_model.features[6].conv[1][0](self.original_model.features[6].conv[0](features[6])))
        t2 = self.original_model.features[13].conv[1][1](self.original_model.features[13].conv[1][0](self.original_model.features[13].conv[0](features[13])))
        t3 = self.original_model.features[18][1](self.original_model.features[18][0](features[18]))
        
        return [t0, t1, t2, t3], skip_feat


class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat)

    def get_bn_before_relu(self):
        BNs = self.encoder.get_bn_before_relu()
        #BNs += self.decoder.get_bn_before_relu()
        return BNs

    def get_channel_num(self):
        channels = self.encoder.get_channel_num()
        #channels += self.decoeder.get_channel_num()
        return channels

    def extract_feature(self, input):
        feats, x = self.encoder.extract_feature(input)
        x = self.decoder(x)
        #feat, x = self.decoder.extract_feature(x)
        #feats += feat
        #x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return feats, x
    
    def get_1x_lr_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

                