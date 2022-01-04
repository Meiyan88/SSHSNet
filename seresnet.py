# encoding:utf-8
# Modify from torchvision
# ResNeXt: Copy from https://github.com/last-one/tools/blob/master/pytorch/SE-ResNeXt/SeResNeXt.py
import torch.nn as nn
import math
from Deeplabv3decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.deeplabv3 import DeepLabV3Plus
import torch.nn.functional as F
import torch
import os
from Deeplabv3decoder import ASPP, SeparableConv3d


class DeepLabV3Plus_2d(DeepLabV3Plus):
    # def __init__(self, encoder_name='se_resnext50_32x4d', classes=20,
    #         encoder_depth = 5,
    #         encoder_weights = "imagenet",
    #         encoder_output_stride= 16,
    #         decoder_channels = 256,
    #         decoder_atrous_rates= (12, 24, 36),
    #         in_channels = 3,
    #         activation = None,
    #         upsampling = 4,
    #         aux_params = None):
    #     super(DeepLabV3Plus_, self).__init__(encoder_name=encoder_name, classes=classes, encoder_depth = encoder_depth,
    #         encoder_weights = encoder_weights,
    #         encoder_output_stride= encoder_output_stride,
    #         decoder_channels = decoder_channels,
    #         decoder_atrous_rates= decoder_atrous_rates,
    #         in_channels = in_channels,
    #         activation = activation,
    #         upsampling = upsampling,
    #         aux_params = aux_params)


    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels, decoder_output

        return masks, decoder_output



class ConvDropoutNormNonlin(nn.Module):
    def __init__(self,inplane, outplane, kernel_size=3, stride=1, padding=1):
        super(ConvDropoutNormNonlin,self).__init__()
        self.conv = nn.Conv3d(inplane, outplane, kernel_size=kernel_size,stride=stride, padding=padding)
        self.instnorm = nn.InstanceNorm3d(outplane,affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
    def forward(self, x):
        return self.lrelu(self.instnorm(self.conv(x)))

class StackedConvLayers(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3, stride=2, padding=1, p=0.0,is_first=False,is_up=False,is_one=True, is_same_last =True):
        super(StackedConvLayers, self).__init__()
        if is_same_last:
            neark_kernel_size = kernel_size
            near_pad = padding
        else:
            neark_kernel_size = 3
            near_pad = 1

        if not is_up:
            if  is_first:
                self.blocks = nn.Sequential(
                    ConvDropoutNormNonlin(inplane, outplane, kernel_size=kernel_size, stride=1, padding=padding),
                    ConvDropoutNormNonlin(outplane, outplane, kernel_size=kernel_size, stride=1, padding=padding)
                )
            else:
                self.blocks = nn.Sequential(
                    ConvDropoutNormNonlin(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding),
                    ConvDropoutNormNonlin(outplane, outplane, kernel_size=neark_kernel_size, stride=1, padding=near_pad)
                )
        else:
            if is_one:
                self.blocks = nn.Sequential(
                    ConvDropoutNormNonlin(inplane, outplane, kernel_size=kernel_size, stride=1, padding=padding),
                )
            else:
                self.blocks = nn.Sequential(
                    ConvDropoutNormNonlin(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding),
                )
    def forward(self,x):
        return self.blocks(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, kernel=(3, 3, 1), padding=(1, 1, 0), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel, stride=stride,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # SE
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_down = nn.Conv3d(
            planes * 4, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv3d(
            planes // 4, planes * 4, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        # Downsample
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        res = out1 * out + residual
        res = self.relu(res)

        return res


class SEResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv3d(21, 64, kernel_size=(7, 7, 1), stride=(2, 2, 1), padding=(3, 3, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 2, 1))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2, 1))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, kernel=3, padding=1)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] *  m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel=(3, 3, 1), padding=(1, 1, 0)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, kernel, padding, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel=kernel, padding=padding))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feature.append(x)
        x = self.layer1(x)
        feature.append(x)
        x = self.layer2(x)
        feature.append(x)
        x = self.layer3(x)
        feature.append(x)
        x = self.layer4(x)
        feature.append(x)

        return feature


class DeeplabV3Plus3D(nn.Module):

    def __init__(self, path, num_classes, encoder='se_resnet50'):
        super(DeeplabV3Plus3D, self).__init__()
        if encoder == 'se_resnet50':
            self.encoder = se_resnet50().cuda(0)
        elif encoder == 'se_resnetx50':
            self.encoder = se_resnext50().cuda(0)
        self.assp = DeepLabV3PlusDecoder(encoder_channels=[64, 256, 512, 1024, 2048]).cuda(1)
        with torch.no_grad():
            #, aux_params={'classes':62}
            if encoder == 'se_resnet50':
                self.net2d = DeepLabV3Plus_2d(encoder_name='se_resnext50_32x4d', classes=20).cuda(1)
            elif encoder == 'se_resnetx50':
                self.net2d = DeepLabV3Plus_2d(encoder_name='se_resnext50_32x4d', classes=20).cuda(1)
            self.load_GPUS(self.net2d, path)
        self.conv2 = nn.Conv3d(512, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=1, bias=False).cuda(1)
        self.bn2 = nn.BatchNorm3d(256).cuda(1)
        self.relu = nn.ReLU(inplace=True)
        self.segmentation = nn.Conv3d(256, num_classes, kernel_size=1).cuda(1)

    def forward(self, x, f, params):
        size = (x.size(2), x.size(3), x.size(4))
        x = x.cuda(0)
        f = f.cuda(1)
        with torch.no_grad():
            m, feature2d = self.net2d(f)
            feature2d = feature2d.permute(dims=[1, 2, 3, 0])
            m = torch.softmax(m, dim=1)
            m = m.permute(dims=[1, 2, 3, 0])
            locfeature = []
            locscore = []
            for i, loc in enumerate(params):
                locfeature.append(torch.unsqueeze(feature2d[:, loc['x_min'] // 4  : loc['x_max'] // 4, loc['y_min'] // 4: loc['y_max'] // 4], dim=0))
                locscore.append(torch.unsqueeze(m[:, loc['x_min']  : loc['x_max'], loc['y_min']: loc['y_max']], dim=0))
            locfeature = torch.cat(locfeature, dim=0)
            locscore = torch.cat(locscore, dim=0)

        x = torch.cat([x, locscore.cuda(0)], dim=1)
        feature = self.encoder(x)
        for i in range(len(feature)):
            feature[i] = feature[i].cuda(1)

        feature3d = self.assp(*feature)

        feature = torch.cat([locfeature.cuda(1) , feature3d], dim=1)
        feature = self.conv2(feature)
        feature = self.bn2(feature)
        feature = self.relu(feature)
        x = self.segmentation(feature)
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True)
        return x

    def load_GPUS(self, model, path):
        file = os.listdir(path)

        state_dict = torch.load(os.path.join(path, file[0]),
                                            map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model



class DeepLabV3Plus_double_out(DeepLabV3Plus):
    def __init__(self, encoder_name='se_resnext50_32x4d', classes=20,
            encoder_depth = 5,
            encoder_weights = "imagenet",
            encoder_output_stride= 16,
            decoder_channels = 256,
            decoder_atrous_rates= (12, 24, 36),
            in_channels = 3,
            activation = None,
            upsampling = 4,
            aux_params = None):
        super(DeepLabV3Plus_double_out, self).__init__(encoder_name=encoder_name, classes=classes, encoder_depth = encoder_depth,
            encoder_weights = encoder_weights,
            encoder_output_stride= encoder_output_stride,
            decoder_channels = decoder_channels,
            decoder_atrous_rates= decoder_atrous_rates,
            in_channels = in_channels,
            activation = activation,
            upsampling = upsampling,
            aux_params = aux_params)


    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels, decoder_output

        return masks ,decoder_output



class DeeplabV3Plus3D_test(nn.Module):
    def __init__(self, path, num_classes, encoder='se_resnet50'):
        super(DeeplabV3Plus3D_test, self).__init__()
        self.encode = encoder
        if encoder == 'se_resnet50' or encoder == 'se_resnetx50_aux':
            self.encoder = se_resnet50().cuda(0)
        elif encoder == 'se_resnetx50':
            self.encoder = se_resnext50().cuda(0)
        self.assp = DeepLabV3PlusDecoder(encoder_channels=[64, 256, 512, 1024, 2048]).cuda(0)
        with torch.no_grad():
            #, aux_params={'classes':62}
            if encoder == 'se_resnet50':
                self.net2d = DeepLabV3Plus_double_out(encoder_name='se_resnet50', classes=20).cuda(1)
            elif encoder == 'se_resnetx50':
                self.net2d = DeepLabV3Plus_double_out(encoder_name='se_resnext50_32x4d', classes=20).cuda(1)
            elif encoder == 'se_resnetx50_aux':
                self.net2d = DeepLabV3Plus_double_out(encoder_name='se_resnet50', classes=20, aux_params={'classes': 62}).cuda(1)

            self.load_GPUS(self.net2d, path)
        self.conv2 = nn.Conv3d(512, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=1, bias=False).cuda(1)
        self.bn2 = nn.BatchNorm3d(256).cuda(1)
        self.relu = nn.ReLU(inplace=True)
        self.segmentation = nn.Conv3d(256, num_classes, kernel_size=1).cuda(1)

    def forward(self, x, f, params):
        size = (x.size(2), x.size(3), x.size(4))
        x = x.cuda(0)
        f = f.cuda(1)
        with torch.no_grad():
            if self.encode == 'se_resnetx50_aux':
                m,_,  feature2d = self.net2d(f)
            else:
                m, feature2d = self.net2d(f)
            feature2d = feature2d.permute(dims=[1, 2, 3, 0])
            m = torch.softmax(m, dim=1)
            m = m.permute(dims=[1, 2, 3, 0])
            locfeature = []
            locscore = []
            for i, loc in enumerate(params):
                locfeature.append(torch.unsqueeze(feature2d[:, loc['x_min'] // 4  : loc['x_max'] // 4, loc['y_min'] // 4: loc['y_max'] // 4], dim=0))
                locscore.append(torch.unsqueeze(m[:, loc['x_min']  : loc['x_max'], loc['y_min']: loc['y_max']], dim=0))
            locfeature = torch.cat(locfeature, dim=0)
            locscore = torch.cat(locscore, dim=0)

        x = torch.cat([x, locscore.cuda(0)], dim=1)
        feature = self.encoder(x)

        feature3d = self.assp(*feature)
        feature = torch.cat([locfeature.cuda(1) , feature3d.cuda(1)], dim=1)
        feature = self.conv2(feature)
        feature = self.bn2(feature)
        feature = self.relu(feature)
        x = self.segmentation(feature)
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True)
        return x

    def load_GPUS(self, model, path):
        file = os.listdir(path)

        state_dict = torch.load(os.path.join(path, file[0]),
                                            map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model

class nnUnet(nn.Module):
    def __init__(self, channel, numclass, path=None, p=0.0, encoder='se_resnetx50'):
        super(nnUnet, self).__init__()
        with torch.no_grad():
            # , aux_params={'classes':62}
            if encoder == 'se_resnet50':
                self.net2d = DeepLabV3Plus_2d(encoder_name='se_resnext50_32x4d', classes=20).cuda(1)
            elif encoder == 'se_resnetx50':
                self.net2d = DeepLabV3Plus_2d(encoder_name='se_resnext50_32x4d', classes=20).cuda(1)
            if path is not None:
                self.load_GPUS(self.net2d, path)


        filters = [32, 64, 128, 256, 320, 320]
        upfilters = [640, 512, 512, 128, 64]
        mapupfilters = [320, 256, 128, 64, 32]
        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.tu = []
        self.seg_outputs = []
        for i in range(len(filters)):
            if i == 0:
                self.conv_blocks_context.append(StackedConvLayers(channel, filters[i], kernel_size=(3, 3, 1), padding=(1, 1, 0),  is_first=True))
            elif i == len(filters) - 1:
                self.conv_blocks_context.append(
                    nn.Sequential(StackedConvLayers(filters[i - 1], filters[i], stride=1, is_up=True, is_one=False),
                                  StackedConvLayers(filters[i - 1], filters[i], is_up=True)))

            elif i < len(filters) - 2:
                self.conv_blocks_context.append(
                    StackedConvLayers(filters[i - 1], filters[i],kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)))

            elif i ==  len(filters) - 2:
                self.conv_blocks_context.append(
                    StackedConvLayers(filters[i - 1], filters[i], kernel_size=(3, 3, 1), stride=(2, 2, 1),
                                      padding=(1, 1, 0), is_same_last=False))

        for i in range(len(upfilters)):
            if i == 0:
                self.conv_blocks_localization.append(
                    nn.Sequential(StackedConvLayers(upfilters[i], mapupfilters[i], is_first=True, is_up=True),
                                  StackedConvLayers(mapupfilters[i], mapupfilters[i], is_first=True, is_up=True)))
            else:
                self.conv_blocks_localization.append(
                    nn.Sequential(StackedConvLayers(upfilters[i], mapupfilters[i], kernel_size=(3, 3, 1), padding=(1, 1, 0), is_first=True, is_up=True),
                                  StackedConvLayers(mapupfilters[i], mapupfilters[i],  kernel_size=(3, 3, 1), padding=(1, 1, 0), is_first=True, is_up=True)))

        for i in range(len(mapupfilters)):
            if i == 0:
                self.tu.append(
                    nn.ConvTranspose3d(mapupfilters[i], mapupfilters[i], kernel_size=1, stride=1, bias=False))
            else:
                self.tu.append(
                    nn.ConvTranspose3d(mapupfilters[i - 1], mapupfilters[i], kernel_size=(2, 2, 1), stride=(2, 2, 1), bias=False))
        for i in range(len(mapupfilters)):
            self.seg_outputs.append(nn.Conv3d(mapupfilters[i], numclass, kernel_size=1, stride=1, bias=False))

        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization).cuda(0)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context).cuda(0)
        self.tu = nn.ModuleList(self.tu).cuda(0)
        self.seg_outputs = nn.ModuleList(self.seg_outputs).cuda(0)

    def forward(self, x, f, params):
        x = x.cuda(0)
        f = f.cuda(1)
        with torch.no_grad():
            m, feature2d = self.net2d(f)
            feature2d = feature2d.permute(dims=[1, 2, 3, 0])
            m = torch.softmax(m, dim=1)
            m = m.permute(dims=[1, 2, 3, 0])
            locfeature = []
            locscore = []
            for i, loc in enumerate(params):
                locfeature.append(torch.unsqueeze(
                    feature2d[:, loc['x_min'] // 4: loc['x_max'] // 4, loc['y_min'] // 4: loc['y_max'] // 4], dim=0))
                locscore.append(torch.unsqueeze(m[:, loc['x_min']: loc['x_max'], loc['y_min']: loc['y_max']], dim=0))
            locfeature = torch.cat(locfeature, dim=0).cuda(0)
            locscore = torch.cat(locscore, dim=0).cuda(0)

        x = torch.cat([x, locscore], dim=1)

        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            if u ==2:
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)], locfeature), dim=1)
            else:
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.seg_outputs[u](x))
        return seg_outputs[-1]

    def load_GPUS(self, model, path):
        file = os.listdir(path)

        state_dict = torch.load(os.path.join(path, file[0]),
                                map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model


class Unet3D(nn.Module):
    def __init__(self, path, num_classes, encoder='se_resnetx50'):
        super(Unet3D, self).__init__()
        if encoder == 'se_resnet50':
            self.encoder = se_resnet50().cuda(0)
        elif encoder == 'se_resnetx50':
            self.encoder = se_resnext50(depth=5).cuda(0)
        filters = [64, 256, 512, 1024, 2048]
        catfilter = [2048, 1024,768, 128, 64]
        project = [  512,  256, 64, 64, 64]
        upfilter = [1024,  512, 256, 64, 64]
        self.aspp = nn.Sequential(
            ASPP(filters[-1], filters[-2], (12, 24, 36), separable=True),
            SeparableConv3d(filters[-2], filters[-2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(filters[-2]),
            nn.ReLU(),
        ).cuda(0)
        self.conv_blocks_localization = []
        self.tu = []
        self.seg_outputs = []
        with torch.no_grad():
            # , aux_params={'classes':62}
            if encoder == 'se_resnet50':
                self.net2d = DeepLabV3Plus_2d(encoder_name='se_resnext50_32x4d', classes=20).cuda(1)
            elif encoder == 'se_resnetx50':
                self.net2d = DeepLabV3Plus_2d(encoder_name='se_resnext50_32x4d', classes=20).cuda(1)
            self.load_GPUS(self.net2d, path)
        for i in range(len(catfilter)):
            if i == 0:
                self.conv_blocks_localization.append(
                    nn.Sequential(StackedConvLayers(catfilter[i], project[i],kernel_size=1, stride=1, padding=0, is_first=True,is_up=True),
                                  StackedConvLayers(project[i], project[i], is_first=True,is_up=True)))
            else:
                self.conv_blocks_localization.append(
                    nn.Sequential(StackedConvLayers(catfilter[i], project[i],  kernel_size=1, stride=1, padding=0, is_first=True, is_up=True),
                                  StackedConvLayers(project[i], project[i], kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), is_first=True, is_up=True)))

        for i in range(len(upfilter)):
            if i ==0:
                self.tu.append(nn.ConvTranspose3d(upfilter[i],upfilter[i], kernel_size=2,stride=2, bias=False))
            else:
                self.tu.append(
                    nn.ConvTranspose3d(upfilter[i], upfilter[i], kernel_size=(2, 2, 1), stride=(2, 2, 1), bias=False))
        self.seg_outputs = nn.Conv3d(in_channels=project[-1],out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=False).cuda(0)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization).cuda(0)
        self.tu = nn.ModuleList(self.tu).cuda(0)

    def forward(self, x, f, params):
        x = x.cuda(0)
        f = f.cuda(1)
        with torch.no_grad():
            m, feature2d = self.net2d(f)
            feature2d = feature2d.permute(dims=[1, 2, 3, 0])
            m = torch.softmax(m, dim=1)
            m = m.permute(dims=[1, 2, 3, 0])
            locfeature = []
            locscore = []
            for i, loc in enumerate(params):
                locfeature.append(torch.unsqueeze(
                    feature2d[:, loc['x_min'] // 4: loc['x_max'] // 4, loc['y_min'] // 4: loc['y_max'] // 4], dim=0))
                locscore.append(torch.unsqueeze(m[:, loc['x_min']: loc['x_max'], loc['y_min']: loc['y_max']], dim=0))
            locfeature = torch.cat(locfeature, dim=0).cuda(0)
            locscore = torch.cat(locscore, dim=0).cuda(0)

        x = torch.cat([x, locscore], dim=1)
        skips = self.encoder(x)
        x = self.aspp(skips[-1])
        for u in range(len(self.tu) - 1):
            if u == 2 :
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 2)], locfeature), dim=1)
                x = self.conv_blocks_localization[u](x)
            else:
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 2)]), dim=1)
                x = self.conv_blocks_localization[u](x)
            # seg_outputs.append(self.seg_outputs[u](x))
        x = self.tu[-1](x)
        x = self.conv_blocks_localization[-1](x)
        x = self.seg_outputs(x)
        return x

    def load_GPUS(self, model, path):
        file = os.listdir(path)

        state_dict = torch.load(os.path.join(path, file[0]),
                                map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model

class Selayer(nn.Module):

    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


class BottleneckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, stride=1, kernel=(3, 3, 1), padding=(1, 1, 0), downsample=None):
        super(BottleneckX, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes * 2)

        self.conv2 = nn.Conv3d(planes * 2, planes * 2, kernel_size=kernel, stride=stride,
                               padding=padding, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(planes * 2)

        self.conv3 = nn.Conv3d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)

        self.selayer = Selayer(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class SEResNeXt(nn.Module):

    def __init__(self, block, layers, input_channel=21, cardinality=32,depth=4):
        super(SEResNeXt, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64

        self.conv1 = nn.Conv3d(input_channel, 64, kernel_size=(7, 7, 1), stride=(2, 2, 1), padding=(3, 3, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 2, 1))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2, 1))
        if depth == 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, kernel=3, padding=1)
        elif depth == 5:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, kernel=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] *  m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel=(3, 3, 1), padding=(1, 1, 0)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, kernel, padding, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, kernel=kernel, padding=padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        feature = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feature.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feature.append(x)
        x = self.layer2(x)
        feature.append(x)
        x = self.layer3(x)
        feature.append(x)
        x = self.layer4(x)
        feature.append(x)

        return feature

def se_resnet50(**kwargs):
    """Constructs a SE-ResNet-50 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet101(**kwargs):
    """Constructs a SE-ResNet-101 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def se_resnet152(**kwargs):
    """Constructs a SE-ResNet-152 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def se_resnext50(**kwargs):
    """Constructs a SE-ResNeXt-50 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNeXt(BottleneckX, [3, 4, 6, 3], **kwargs)
    return model


def se_resnext101(**kwargs):
    """Constructs a SE-ResNeXt-101 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNeXt(BottleneckX, [3, 4, 23, 3], **kwargs)
    return model


def se_resnext152(**kwargs):
    """Constructs a SE-ResNeXt-152 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNeXt(BottleneckX, [3, 8, 36, 3], **kwargs)
    return model

# import torch
# torch.cuda.set_device(6)
# x = torch.ones((4, 21, 192, 192, 12)).cuda()
#
# # x = torch.ones((2, 1, 192, 192, 12)).cuda()
# # h = torch.ones((12, 3, 880, 880)).cuda()
# model = nnUnet(channel=21, numclass=20).cuda()
# model(x)