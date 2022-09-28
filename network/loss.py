# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from network.net_ops import VGG19


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.MSELoss()
        self.transform = nn.functional.interpolate
    
    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        # print('resized shape:{}'.format(input.size()))
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        
        loss = self.criterion(input_features[3], target_features[3])
        
        return loss


class CosineOrthogonalLoss(nn.Module):
    # flow shape [N,H,W,2]
    def __init__(self, dim=3):
        super(CosineOrthogonalLoss, self).__init__()
        self.criterion = torch.nn.CosineSimilarity(dim=dim)

    def forward(self, src, target):
        output = self.criterion(src, target).abs()
        output[output < 0.5] = 0  # penalize flows whose cosine bigger than 0.5
        return output.mean()


class TVLoss(nn.Module):
    #  [N,C,H,W]
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight
    
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()

        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()
    
    def forward(self, x):
        batch_size, slice_num = x.size()[:2]
        z_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]
        count_z = self._tensor_size(x[:, :, 1:, :, :])
        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])
        z_tv = torch.pow((x[:, :, 1:, :, :] - x[:, :, :z_x - 1, :, :]), 2).sum()
        h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1]), 2).sum()
        return 2 * (z_tv / count_z + h_tv / count_h + w_tv / count_w) / (batch_size * slice_num)
    
    def _tensor_size(self, t):
        return t.size()[2] * t.size()[3] * t.size()[4]


class LaplacianLoss_L1(nn.Module):
    def __init__(self):
        super(LaplacianLoss_L1, self).__init__()
    
    def forward(self, x):
        batch_size, slice_num = x.size()[:2]
        z_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]
        count_z = self._tensor_size(x[:, :, 1:, :, :])
        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])
        z_tv = torch.abs((x[:, :, 1:, :, :] - x[:, :, :z_x - 1, :, :])).sum()
        h_tv = torch.abs((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :])).sum()
        w_tv = torch.abs((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1])).sum()
        return 2 * (z_tv / count_z + h_tv / count_h + w_tv / count_w) / (batch_size * slice_num)
    
    def _tensor_size(self, t):
        return t.size()[2] * t.size()[3] * t.size()[4]



class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)
