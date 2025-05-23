import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
from collections import OrderedDict
import math

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

class Update_mask(nn.Module):
    def __init__(self, feat_channels):
        super(Update_mask, self).__init__()

        self.conv0 = nn.Conv2d(1, feat_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(feat_channels[0], feat_channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(feat_channels[1], feat_channels[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(feat_channels[2], feat_channels[3], kernel_size=3, stride=2, padding=1, bias=False)
        '''
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, feat_channels[0], kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(feat_channels[0])),
            ('relu0', nn.ReLU(inplace=True))
        ]))
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(feat_channels[0], feat_channels[1], kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(feat_channels[1])),
            ('relu0', nn.ReLU(inplace=True))
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(feat_channels[1], feat_channels[2], kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(feat_channels[2])),
            ('relu0', nn.ReLU(inplace=True))
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(feat_channels[2], feat_channels[3], kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(feat_channels[3])),
            ('relu0', nn.ReLU(inplace=True))
        ]))
        '''
        
    def forward(self, mask):
        mask0 = self.conv0(mask)
        mask1 = self.conv1(mask0)
        mask2 = self.conv2(mask1)
        mask3 = self.conv3(mask2)

        return [mask0, mask1, mask2, mask3]

class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        t_channels = t_net.module.get_channel_num()
        s_channels = s_net.module.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])
        self.update_mask = Update_mask(t_channels)

        teacher_bns = t_net.module.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net


    def forward(self, x, x_NT='none', mask='none'):
        if x_NT=='none':
            t_feats, t_out = self.t_net.module.extract_feature(x)
        else:    
            t_feats, t_out = self.t_net.module.extract_feature(x_NT)
        s_feats, s_out = self.s_net.module.extract_feature(x)
        feat_num = len(t_feats)

        loss_distill = 0

        if mask=='none':
            for i in range(feat_num):
                s_feats[i] = self.Connectors[i](s_feats[i])
                loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                                / 2 ** (feat_num - i - 1)
        else:
            mask_up = self.update_mask(mask)
            for i in range(feat_num):
                s_feats[i] = self.Connectors[i](s_feats[i])
                loss_distill += distillation_loss(s_feats[i]*mask_up[i], t_feats[i]*mask_up[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                                / 2 ** (feat_num - i - 1)

        return s_out, loss_distill
        # return s_out[0], s_out[1], s_out[2], s_out[3], s_out[4]