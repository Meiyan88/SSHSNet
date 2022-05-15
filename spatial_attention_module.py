import torch
import math
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt

# from pyheatmap
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = (7, 7, 1)
        self.compress = ChannelPool()
        # self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(3, 3, 0), relu=False)

    def forward(self, x2d, x3d):
        x_compress = self.compress(x2d)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting

        # ---------------------imaging-------------------------
        # # sns.heatmap(scale.cpu().detach().numpy())
        # # img = f.cpu().numpy()
        # # p1 = scale.cpu().detach().numpy()
        #
        # # plt.subplot(121)
        # # plt.imshow(p1[0, 0,:,:,5])
        # # plt.subplot(122)
        # out = x3d*scale
        # out = out.cpu().detach().numpy()
        # plt.imshow(out[0, 0,:,:,5])
        # # plt.subplot(133)
        # # plt.imshow(img[:, :, 6])
        # plt.show()
        # ---------------------imaging-------------------------

        return  scale *  x3d


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv =  nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv =  nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma =  nn.Parameter(torch.zeros(1))

        self.softmax =  nn.Softmax(dim=-1)
    def forward(self, x2d, x3d):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        orix3d = x3d
        x2d = x2d.permute(0, 1, 4, 2, 3)
        x3d = x3d.permute(0, 1, 4, 2, 3)
        m_batchsize, c,  depth, height, width  = x2d.size()
        # m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x2d).view(m_batchsize, -1,  width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x2d).view(m_batchsize, -1,  width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x2d).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize,  c,  depth, height, width).permute(0, 1, 3, 4, 2)

        out = self.gamma*out + orix3d
        return out

# class CAM_Module(nn.Module):
#     """ Channel attention module"""
#     def __init__(self, in_dim, d=1):
#         super(CAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         self.d = d
#         self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // self.d , kernel_size=1)
#         self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // self.d , kernel_size=1)
#         self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax  = nn.Softmax(dim=-1)
#     def forward(self,x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         m_batchsize, C, height, width, depth= x.size()
#         x = x.contiguous()
#
#         proj_query = self.query_conv(x).view(m_batchsize, C // self.d , -1)
#         proj_key = self.key_conv(x).view(m_batchsize, C // self.d , -1).permute(0, 2, 1)
#         energy = torch.bmm(proj_query, proj_key)
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
#         attention = self.softmax(energy_new)
#         proj_value = self.value_conv(x).view(m_batchsize, C, -1)
#
#         out = torch.bmm(attention, proj_value)
#         out = out.view(m_batchsize, C, height, width, depth)
#
#         out = self.gamma*out + x
#         return out



class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_channel=True):
        super(CBAM, self).__init__()
        self.SpatialGate = PAM_Module(gate_channels)
        self.no_channel = no_channel
        if not self.no_channel:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)

    def forward(self, x2d, x3d):
        # if not self.no_channel:
        #     x = self.ChannelGate(x)
        x_out = self.SpatialGate(x2d, x3d)
        return x_out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        # self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, depth, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, depth, height, width)

        out = self.gamma * out + x
        return out


class z_axis_Gate(nn.Module):
    def __init__(self, in_dim, ):
        super(z_axis_Gate, self).__init__()
        self.size = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        # self.relu = nn.ReLU()
    def forward(self, x_2d, x_3d):

        ori2dx = x_2d

        x_2d = x_2d.permute(0, 4, 1, 2, 3)
        x_3d = x_3d.permute(0, 4, 1, 2, 3)

        m_batchsize, depth, c, height, width  = x_3d.size()

        proj_query = x_3d.view(m_batchsize, depth, -1) ## cuole1
        proj_key = x_3d.view(m_batchsize, depth, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        z_att = self.softmax(energy_new)
        # z_att = self.relu(energy_new)
        m_batchsize,  depth, c, height, width = x_2d.size()
        proj_value = x_3d.view(m_batchsize, depth, -1)
        out = torch.bmm(z_att, proj_value)
        # x_2d=z_att*
        out = out.view(m_batchsize, depth, c, height, width)

        out = self.gamma * out.permute(0, 2, 3, 4, 1) + ori2dx
        # out = self.gamma * out
        # ---------------------------------
        # p2 = out.cpu().detach().numpy()
        # plt.imshow(p2[0, 0, :, :, 5])
        # plt.show()
        # ---------------------------
        return out


class DualCrossAttModule_2(nn.Module):
    def __init__(self, gate_channels, no_channel=True, in_dim=12):
        # def __init__(self, gate_channels1, gate_channels, no_channel=True, in_dim=12,in_planes=20,out_planes=1):
        super(DualCrossAttModule_2, self).__init__()
        # self.sam1 = CBAM(gate_channels=gate_channels1, no_channel=no_channel)
        self.sam = CBAM(gate_channels=gate_channels, no_channel=no_channel)
        self.za = z_axis_Gate(in_dim=in_dim)
        # self.conv = nn.Conv3d(in_planes,out_planes,kernel_size=(1,1,1),stride=1, bias=False)

    def forward(self, x_2d, x_3d):
        # GIVE spital to 3D
        x_3d_sa = self.sam(x_2d, x_3d)
        # x_2d_sa_conv = self.conv(x_2d_sa)
        # x_3d_za = torch.mul(x_2d_sa_conv, x_3d)

        # GIVE Z　信息给 2D
        x_2d_za = self.za(x_2d, x_3d_sa)
        x = torch.cat([x_2d_za, x_3d_sa], dim=1)
        return x

class DualCrossAttModule(nn.Module):
    def __init__(self, gate_channels=10, no_channel=True, in_dim=12):
        # def __init__(self, gate_channels1, gate_channels, no_channel=True, in_dim=12,in_planes=20,out_planes=1):
        super(DualCrossAttModule, self).__init__()
        # self.sam1 = CBAM(gate_channels=gate_channels1, no_channel=no_channel)
        self.sam = CBAM(gate_channels=gate_channels, no_channel=no_channel)
        self.za = z_axis_Gate(in_dim=in_dim)
        # self.conv = nn.Conv3d(in_planes,out_planes,kernel_size=(1,1,1),stride=1, bias=False)

    def forward(self, x_2d, x_3d):
        # GIVE spital to 3D
        x_3d_sa = self.sam(x_2d, x_3d)

        # GIVE Z　信息给 2D
        x_2d_za = self.za(x_2d, x_3d)

        x = x_2d_za +  x_3d_sa
        return x


class TrippleDualCrossAttModule(nn.Module):
    def __init__(self, gate_channels=10, channle = 512, no_channel=True, in_dim=12, mode= 'cat'):
        # def __init__(self, gate_channels1, gate_channels, no_channel=True, in_dim=12,in_planes=20,out_planes=1):
        super(TrippleDualCrossAttModule, self).__init__()
        # self.sam1 = CBAM(gate_channels=gate_channels1, no_channel=no_channel)
        self.sam = CBAM(gate_channels=gate_channels, no_channel=no_channel)
        self.za = z_axis_Gate(in_dim=in_dim)
        if mode == 'sum':
            channle = channle // 2
        self.chan = CAM_Module(channle)
        self.mode = mode
        # self.conv = nn.Conv3d(in_planes,out_planes,kernel_size=(1,1,1),stride=1, bias=False)

    def forward(self, x_2d, x_3d):
        # GIVE spital to 3D
        x_3d_sa = self.sam(x_2d, x_3d)

        # GIVE Z　信息给 2D
        x_2d_za = self.za(x_2d, x_3d)

        if self.mode == 'cat':
            x = torch.cat([x_2d_za,  x_3d_sa],dim=1)
        elif self.mode == 'sum':
            x = x_2d_za + x_3d_sa

        # channel attention
        x = self.chan(x)
        return x
