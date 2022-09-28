# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_layer, self).__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SASA(nn.Module):
    '''
        Structure Affinity Self attention Module
    '''

    def __init__(self, in_dim):
        super(SASA, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.mag_conv = nn.Conv2d(in_channels=5, out_channels=in_dim//32, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
        self.sigmoid = nn.Sigmoid()

    def structure_encoder(self, paf_mag, target_height, target_width):
        torso_mask = torch.sum(paf_mag[:, 1:3, :, :], dim=1, keepdim=True)
        torso_mask = torch.clamp(torso_mask, 0, 1)

        arms_mask = torch.sum(paf_mag[:, 4:8, :, :], dim=1, keepdim=True)
        arms_mask = torch.clamp(arms_mask, 0, 1)

        legs_mask = torch.sum(paf_mag[:, 8:12, :, :], dim=1, keepdim=True)
        legs_mask = torch.clamp(legs_mask, 0, 1)

        fg_mask = paf_mag[:, 12, :, :].unsqueeze(1)
        bg_mask = 1 - fg_mask
        Y = torch.cat((arms_mask, torso_mask, legs_mask, fg_mask, bg_mask), dim=1)
        Y = F.interpolate(Y, size=(target_height, target_width), mode="area")
        return Y


    def forward(self, X, PAF_mag):
        """
            inputs :
                x : input feature maps( B x C x H x W)
                Y : ( B x C x H x W), 1 denotes connectivity, 0 denotes non-connectivity
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, height, width = X.size()

        # PAF_mag = PAF_mag.contiguous()

        Y = self.structure_encoder(PAF_mag, height, width)

        connectivity_mask_vec = self.mag_conv(Y).view(m_batchsize, -1, width * height)  # B * C * (W*H)
        affinity = torch.bmm(connectivity_mask_vec.permute(0, 2, 1),connectivity_mask_vec)  # B * (W*H) * (W*H)
        affinity_centered = affinity - torch.mean(affinity) # centering
        affinity_sigmoid = self.sigmoid( affinity_centered)

        proj_query = self.query_conv(X).view(m_batchsize, -1, width * height).permute(0, 2, 1)  #  B * (W*H) * C
        proj_key = self.key_conv(X).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        selfatten_map = torch.bmm(proj_query, proj_key)  # B * (W*H) * (W*H)
        selfatten_centered = selfatten_map - torch.mean(selfatten_map)  # centering
        selfatten_sigmoid = self.sigmoid(selfatten_centered)

        SASA_map = selfatten_sigmoid * affinity_sigmoid

        proj_value = self.value_conv(X).view(m_batchsize, -1, width * height)  # B * C * (W*H)

        out = torch.bmm(proj_value, SASA_map.permute(0, 2, 1)) # B* C *(W*H)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + X
        return out, Y


class FlowGenerator(nn.Module):
    def __init__(self, n_channels, deep_supervision=False):
        super(FlowGenerator, self).__init__()
        self.deep_supervision = deep_supervision

        self.Encoder = nn.Sequential(
            conv_layer(n_channels, 64),
            conv_layer(64, 64),
            nn.MaxPool2d(2),
            conv_layer(64, 128),
            conv_layer(128, 128),
            nn.MaxPool2d(2),
            conv_layer(128, 256),
            conv_layer(256, 256),
            nn.MaxPool2d(2),
            conv_layer(256, 512),
            conv_layer(512, 512),
            nn.MaxPool2d(2),
            conv_layer(512, 1024),
            conv_layer(1024, 1024),
            conv_layer(1024, 1024),
            conv_layer(1024, 1024),
            conv_layer(1024, 1024),
        )

        self.SASA = SASA(in_dim=1024)

        self.Decoder = nn.Sequential(
            conv_layer(1024, 1024),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv_layer(1024, 512),
            conv_layer(512, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv_layer(512, 256),
            conv_layer(256, 256),
            conv_layer(256, 128),
            conv_layer(128, 64),
            conv_layer(64, 32),
            nn.Conv2d(32, 2, kernel_size=1, padding=0),
            nn.Tanh(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )

        dilation_ksize = 17
        self.dilation= torch.nn.MaxPool2d(kernel_size=dilation_ksize, stride=1, padding=int((dilation_ksize - 1) / 2))


    def warp(self, x, flow, mode='bilinear', padding_mode='zeros', coff=0.2):
        n, c, h, w = x.size()
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        xv = xv.float() / (w - 1) * 2.0 - 1
        yv = yv.float() / (h - 1) * 2.0 - 1

        '''
        grid[0,:,:,0] =
        -1, .....1
        -1, .....1
        -1, .....1

        grid[0,:,:,1] =
        -1,  -1, -1
         ;        ;
         1,   1,  1


        image  -1 ~1       -128~128 pixel
        flow   -0.4~0.4     -51.2~51.2 pixel
        '''

        if torch.cuda.is_available():
            grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
        else:
            grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0)
        grid_x = grid + 2 * flow * coff
        warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)
        return warp_x


    def forward(self, img, skeleton_map, coef=0.2):
        '''
        img  -1 ~ 1
        skeleton_map  -1 ~ 1
        '''

        img_concat = torch.cat((img, skeleton_map), dim=1)
        X = self.Encoder(img_concat)

        _, _, height, width = X.size()

        # directly get PAF magnitude from skeleton maps via dilation
        PAF_mag = self.dilation((skeleton_map + 1.0) * 0.5)

        out, Y = self.SASA(X, PAF_mag)
        flow = self.Decoder(out)

        flow = flow.permute(0, 2, 3, 1)  # [n, 2, h, w] ==> [n, h, w, 2]

        warp_x = self.warp(img, flow, coff=coef)
        warp_x = torch.clamp(warp_x, min=-1.0, max=1.0)

        return warp_x, flow


if __name__ == "__main__":
    model = FlowGenerator(n_channels = 16)

    device = torch.device('cuda:0')

    model.to(device)
    in_tensor = torch.rand([4,3,512,512],dtype=torch.float32)
    skeleton_tensor = torch.rand([4,13,512,512],dtype=torch.float32)
    in_tensor = in_tensor.to(device)
    skeleton_tensor = skeleton_tensor.to(device)

    warp, flow = model(in_tensor, skeleton_tensor)
    print('warp shape:{}'.format(warp.shape))
    print('flow shape:{}'.format(flow.shape))
