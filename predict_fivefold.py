import numpy as np
import nibabel as nib
import math
import os
import torch
from segmentation_models_pytorch.deeplabv3 import DeepLabV3Plus
import torch.nn as nn
from skimage.transform import resize
from seresnet import se_resnext50, DeepLabV3PlusDecoder, se_resnet50
import torch.nn.functional as F
from spatial_attention_module import TrippleDualCrossAttModule

def normalize(img):
    data = img
    imin = np.percentile(data,0.1)
    imax = np.percentile(data,99.9)
    data = ((np.clip(data,imin,imax) - imin) * 255 / (imax - imin))
    return data

def fix(ix,mx,length=128,maxx=240):
    dist = mx - ix
    if dist < length:
        if (length - dist) % 2 == 0:
            mx = mx + int((length - dist) / 2)
            ix = ix - int((length - dist) / 2)
        else:
            mx = mx + int((length - dist) / 2) + 1
            ix = ix - int((length - dist) / 2)
    if ix < 0:
        mx = mx + abs(ix)
        ix = 0
    elif mx >= maxx:
        ix = ix - (mx - (maxx - 1))
        mx = maxx - 1
    return ix,mx

def cmp(ix, mx, newix,newmx):
    if ix > newix:
        ix = newix
    if mx < newmx:
        mx = newmx
    return ix,mx

def index(ix,mx,strid= 32):
    dist = mx - ix
    if dist %  strid != 0:
        time = int(dist /strid)
    else:
        time = int(dist / strid) - 1
    ind = []
    for i in range(time):
        if i ==0:
            ind.append(ix)
        elif i == time-1:
            ind.append(mx - 64)
        else:
            ix += 32
            ind.append(ix)
    return ind

def gettimes(new_shape, input_size, strides):
    num_x = 1 + math.ceil((new_shape[0] - input_size[0]) / strides[0])
    num_y = 1 + math.ceil((new_shape[1] - input_size[1]) / strides[1])
    num_z = 1 + math.ceil((new_shape[2] - input_size[2]) / strides[2])
    # print(CT_origianl.shape, change_spacing_shape, new_shape, num_x, num_y, num_z)
    return num_x, num_y, num_z


def load_GPUS(model, path):
    file = os.listdir(path)
    state_dict = torch.load(os.path.join(path, file[0]),
                            map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model




def pad(image, new_shape, border_mode="constant", value=0):
    '''
    image: [H, W, D, C] or [H, W, D]
    new_shape: [H, W, D]
    '''
    axes_not_pad = len(image.shape) - len(new_shape)

    old_shape = np.array(image.shape[:len(new_shape)])
    new_shape = np.array([max(new_shape[i], old_shape[i]) for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference - pad_below

    pad_list = [list(i) for i in zip(pad_below, pad_above)] + [[0, 0]] * axes_not_pad

    if border_mode == 'reflect':
        res = np.pad(image, pad_list, border_mode)
    elif border_mode == 'constant':
        res = np.pad(image, pad_list, border_mode, constant_values=value)
    else:
        raise ValueError

    return res, pad_list

def padupdown(img, axis):
    if axis == 0:
        img = np.concatenate([img[0:1, :, :], img, img[img.shape[0]-1: img.shape[0],:,:]], axis=axis)
    elif axis == -1 or axis == 2:
        img = np.concatenate([img[:, :, 0:1], img, img[:, :, img.shape[axis] - 1: img.shape[axis]]], axis=axis)
    return img


class DeepLabV3Plus_(DeepLabV3Plus):
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
        super(DeepLabV3Plus_, self).__init__(encoder_name=encoder_name, classes=classes, encoder_depth = encoder_depth,
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

        return decoder_output, masks


class DualNetwork(nn.Module):
    def __init__(self, config):
        super(DualNetwork, self).__init__()
        self.config = config
        self.branch1 = DeepLabV3Plus_(encoder_name='se_resnext50_32x4d', classes=20, encoder_weights=None)
        self.branch2 = DeepLabV3Plus_(encoder_name='se_resnext50_32x4d', classes=20, encoder_weights=None)


    def forward(self, f, params):
        feature2db1, mb1 = self.branch1(f)
        feature2db2, mb2 = self.branch2(f)
        feature2db1 = F.interpolate(feature2db1, scale_factor=2, mode='bilinear', align_corners=True)
        feature2db2 = F.interpolate(feature2db2, scale_factor=2, mode='bilinear', align_corners=True)
        feature2db1 = feature2db1.permute(dims=[1, 2, 3, 0])
        feature2db2 = feature2db2.permute(dims=[1, 2, 3, 0])
        mb1 = torch.softmax(mb1, dim=1)
        mb2 = torch.softmax(mb2, dim=1)
        mb1 = F.interpolate(mb1, scale_factor=2, mode='nearest')
        mb2 = F.interpolate(mb2, scale_factor=2, mode='nearest')
        mb1 = mb1.permute(dims=[1, 2, 3, 0])
        mb2 = mb2.permute(dims=[1, 2, 3, 0])
        m = (mb1 + mb2) / 2
        feature2d = torch.cat([feature2db1, feature2db2], dim=0)
        locfeature = []
        locscore = []


        for i, loc in enumerate(params):
            locfeature.append(torch.unsqueeze(
                feature2d[:, loc['x_min'] // 4: loc['x_max'] // 4, loc['y_min'] // 4: loc['y_max'] // 4], dim=0))
            locscore.append(torch.unsqueeze(m[:, loc['x_min']: loc['x_max'], loc['y_min']: loc['y_max']], dim=0))
        locfeature = torch.cat(locfeature, dim=0)
        locscore = torch.cat(locscore, dim=0)
        return locfeature, locscore



class DeeplabV3Plus3D(nn.Module):

    def __init__(self,  num_classes, encoder='se_resnetx50'):
        super(DeeplabV3Plus3D, self).__init__()
        if encoder == 'se_resnet50':
            self.encoder = se_resnet50()
        elif encoder == 'se_resnetx50':
            self.encoder = se_resnext50()
        self.assp = DeepLabV3PlusDecoder(encoder_channels=[64, 256, 512, 1024, 2048])
        self.conv2 = nn.Conv3d(768, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=1, bias=False)
        self.bn2 = nn.BatchNorm3d(256)
        self.relu = nn.ReLU(inplace=True)
        self.segmentation = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, x, locfeature, locscore):
        size = (x.size(2), x.size(3), x.size(4))
        x = torch.cat([x, locscore], dim=1)
        feature = self.encoder(x)
        feature3d = self.assp(*feature)
        feature = torch.cat([locfeature, feature3d], dim=1)
        feature = self.conv2(feature)
        feature = self.bn2(feature)
        feature = self.relu(feature)
        x = self.segmentation(feature)
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True)
        return x


class DeeplabV3Plus3D_tripple(nn.Module):

    def __init__(self,  num_classes, encoder='se_resnetx50', mode='cat'):
        super(DeeplabV3Plus3D_tripple, self).__init__()
        self.mode = mode
        if encoder == 'se_resnet50':
            self.encoder = se_resnet50()
        elif encoder == 'se_resnetx50':
            self.encoder = se_resnext50()
        self.assp = DeepLabV3PlusDecoder(encoder_channels=[64, 256, 512, 1024, 2048])

        self.conv1 = nn.Conv3d(512, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(256)
        self.relu = nn.ReLU(inplace=True)

        if mode == 'cat':
            self.conv2 = nn.Conv3d(512, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=1, bias=False)
            self.bn2 = nn.BatchNorm3d(256)

        self.segmentation = nn.Conv3d(256, num_classes, kernel_size=1)
        # self.dual_att_module_input = DualCrossAttModule(gate_channels=21, no_channel=True)
        self.dual_att_module_feature = TrippleDualCrossAttModule(gate_channels=256, channle=512, no_channel=True, mode=mode)

    def forward(self, x, locfeature, locscore):
        size = (x.size(2), x.size(3), x.size(4))
        # x = self.dual_att_module_input(locscore, x)
        x = torch.cat([x, locscore], dim=1)
        feature = self.encoder(x)
        feature3d = self.assp(*feature)
        locfeature = self.conv1(locfeature)
        locfeature = self.bn1(locfeature)
        locfeature = self.relu(locfeature)

        feature = self.dual_att_module_feature(locfeature, feature3d)

        if self.mode == 'cat':
            feature = self.conv2(feature)
            feature = self.bn2(feature)
            feature = self.relu(feature)

        x = self.segmentation(feature)
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True)
        return x


if __name__ == '__main__':
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,default="0,1", help='use gpu id, defalut use two gpu.')
    parser.add_argument('--branch_best', type=str, default="branch2", help='the best_branch use in 2D network')
    parser.add_argument('--exid2D', type=str, default="ex0", help='2d model save.')
    parser.add_argument('--exid3D', type=str, default="ex1", help='3d model save')
    parser.add_argument('--datapath', type=str, default='./dataset/processdata3D_test', help='data path for predict')
    parser.add_argument('--oridatapath', type=str, default='./test/MR', help='original dataset, to get the original information')
    parser.add_argument('--infomation', type=str, default='info.csv', help='the file name of imformation that generating in processing')
    parser.add_argument('--batch_size', type=int, default=20, help='test batch size')
    config = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu # use two gpus
    div = [ 'sub0', 'sub1', 'sub2', 'sub3', 'sub4']


    savepath = os.path.join(os.path.join(config.datapath, config.exid3D), 'predict')
    os.makedirs(savepath,exist_ok=True)
    df = pd.read_csv(os.path.join(config.datapath, config.infomation))

    allname = df['name'].tolist()
    stridemin = df['stridemin'].values
    stridemax = df['stridemax'].values

    resizez = df['allresolution_size0'].values
    resizey = df['allresolution_size1'].values
    resizex = df['allresolution_size2'].values

    oriz = df['orishape0'].values
    oriy = df['orishape1'].values
    orix = df['orishape2'].values

    filename = os.listdir(os.path.join(config.datapath, 'image'))

## load model
    model2dall = []
    model3dall = []
    for l,file in enumerate(div):
            model2d = DualNetwork(None).cuda(1)
            model2d = load_GPUS(model2d,
                              os.path.join(os.path.join(os.path.join('./weight', config.exid2D), file), config.branch_best))

            model3d = DeeplabV3Plus3D_tripple(num_classes=20,
                                    encoder='se_resnetx50', mode='cat').cuda()
            model3d = nn.DataParallel(model3d, device_ids=[0, 1])
            modelfile = os.listdir(os.path.join(os.path.join('./weight', config.exid3D), file))[0]
            model3d.load_state_dict(torch.load(os.path.join(os.path.join(os.path.join('./weight', config.exid3D), file), modelfile),map_location='cpu'))

            model2dall.append(model2d)
            model3dall.append(model3d)

    ## predict and save sample from five model
    with torch.no_grad():
        for i,name in enumerate(filename):
            inde = allname.index(name)
            imgx = nib.load(os.path.join(os.path.join(config.datapath, 'image'), name))
            imgx =imgx.get_fdata()

            # imgx = padupdown(imgx, axis=-1)
            imgx, padlist = pad(imgx, new_shape=[512, 1024])
            input_size = [512,1024,12]
            strides = [100, 100, 6]
            inner_input_size = [192, 192, 12]
            inner_strides = [81, 81, 1]
            num_x, num_y, num_z = gettimes(imgx.shape, input_size, strides)
            num_x_in, num_y_in, num_z_in = gettimes(input_size, inner_input_size, inner_strides)
            fivemask = []

            for m in range(len(model2dall)):
                putbin = torch.zeros((20, imgx.shape[0], imgx.shape[1], imgx.shape[2]))
                countbinout = torch.zeros([imgx.shape[0], imgx.shape[1], imgx.shape[2]])
                model3dall[m].eval()
                model2dall[m].eval()
                torch.cuda.empty_cache()
                samplemask = []
                countbin = []
                for x in range(num_x):
                    for y in range(num_y):
                        for z in range(num_z):
                            count = 0
                            x_left = x * strides[0]
                            x_right = x * strides[0] + input_size[0]
                            y_up = y * strides[1]
                            y_down = y * strides[1] + input_size[1]
                            z_up = z * strides[2]
                            z_down = z * strides[2] + input_size[2]
                            if x == num_x - 1:
                                x_left = imgx.shape[0] - input_size[0]
                                x_right = imgx.shape[0]
                            if y == num_y - 1:
                                y_up = imgx.shape[1] - input_size[1]
                                y_down = imgx.shape[1]
                            if z == num_z - 1:
                                z_up = imgx.shape[2] - input_size[2]
                                z_down = imgx.shape[2]
                            inpt = imgx[x_left:x_right, y_up:y_down, z_up:z_down]
                            putbininer = torch.zeros([20, 512,1024, 12])
                            countbininer = torch.zeros([512,1024, 12])

                            if z_up == 0:
                                downimg = np.expand_dims(np.concatenate(
                                    [imgx[x_left: x_right, y_up: y_down, 0:1],
                                     imgx[x_left: x_right, y_up: y_down,
                                     0: input_size[-1] - 1]], axis=-1), axis=0)
                            else:
                                downimg = np.expand_dims(
                                    imgx[x_left: x_right, y_up: y_down,
                                    z_up - 1: z_up - 1 + input_size[-1]], axis=0)
                            if z_down == imgx.shape[-1]:
                                upimg = np.expand_dims(np.concatenate([imgx[x_left: x_right, y_up: y_down,
                                                                       imgx.shape[-1] - input_size[-1] + 1: imgx.shape[
                                                                           -1]]
                                                                          , imgx[x_left: x_right, y_up: y_down,
                                                                            imgx.shape[-1] - 1: imgx.shape[-1]]],
                                                                      axis=-1), axis=0)
                            else:
                                upimg = np.expand_dims(
                                    imgx[x_left: x_right, y_up: y_down,
                                    z_down + 1 - input_size[-1]: z_down + 1], axis=0)
                            img_ = np.concatenate([downimg, np.expand_dims(inpt, axis=0), upimg], axis=0)
                            # plt.subplot(121)
                            # plt.imshow(img_[0, :,:,5])
                            img_ = np.transpose(img_, axes=[1, 2, 3, 0])
                            img_ = resize(img_, output_shape=[256, 512, 12], order=1, mode='constant', clip=False, preserve_range=True)
                            img_ = np.transpose(img_, axes=[2, 3, 0, 1])
                            # plt.subplot(122)
                            # plt.imshow(img_[5, 0, :, :])
                            # plt.show()

                            batch = []
                            locdict = []
                            batchmask = []

                            for ix in range(num_x_in):
                                for iy in range(num_y_in):
                                    for iz in range(num_z_in):
                                        x_left_in = ix * inner_strides[0]
                                        x_right_in = ix * inner_strides[0] + inner_input_size[0]
                                        y_up_in = iy * inner_strides[1]
                                        y_down_in = iy * inner_strides[1] + inner_input_size[1]
                                        z_up_in = iz * inner_strides[2]
                                        z_down_in = iz * inner_strides[2] + inner_input_size[2]
                                        if ix == num_x_in - 1:
                                            x_left_in = input_size[0] - inner_input_size[0]
                                            x_right_in = input_size[0]
                                        if iy == num_y_in - 1:
                                            y_up_in = input_size[1] - inner_input_size[1]
                                            y_down_in = input_size[1]
                                        if iz == num_z_in - 1:
                                            z_up_in = input_size[2] - inner_input_size[2]
                                            z_down_in = input_size[2]
                                        inpt_in = inpt[x_left_in:x_right_in, y_up_in:y_down_in, z_up_in:z_down_in]
                                        batch.append(inpt_in)
                                        locdict.append({'x_min':torch.tensor(x_left_in), 'x_max':torch.tensor(x_right_in),'y_min':torch.tensor(y_up_in),
                                                        'y_max':torch.tensor(y_down_in),'z_min':torch.tensor(z_up_in), 'z_max':torch.tensor(z_down_in)})
                                        count += 1
                                        if count % config.batch_size == 0 or count == num_x_in * num_y_in * num_z_in:
                                            batch = np.expand_dims(np.asarray(batch), axis=1)
                                            batch = torch.from_numpy(batch).float().cuda()
                                            iner2d = torch.from_numpy(img_).float().cuda(1)
                                            # batch = batch.permute([0, 3, 1, 2])
                                            locfeature, locscore = model2dall[m](iner2d, locdict)
                                            locfeature = locfeature.cuda()
                                            locscore = locscore.cuda()
                                            temppred = model3dall[m](batch, locfeature,  locscore)
                                            temppred = torch.softmax(temppred, dim=1).cpu()
                                            batchmask.append(temppred)
                                            batch = []
                                            locdict = []

                            batchmask = torch.cat(batchmask, dim=0)
                            loc =0
                            for ix in range(num_x_in):
                                for iy in range(num_y_in):
                                    for iz in range(num_z_in):
                                        x_left_in = ix * inner_strides[0]
                                        x_right_in = ix * inner_strides[0] + inner_input_size[0]
                                        y_up_in = iy * inner_strides[1]
                                        y_down_in = iy * inner_strides[1] + inner_input_size[1]
                                        z_up_in = iz * inner_strides[2]
                                        z_down_in = iz * inner_strides[2] + inner_input_size[2]
                                        if ix == num_x_in - 1:
                                            x_left_in = input_size[0] - inner_input_size[0]
                                            x_right_in = input_size[0]
                                        if iy == num_y_in - 1:
                                            y_up_in = input_size[1] - inner_input_size[1]
                                            y_down_in = input_size[1]
                                        if iz == num_z_in - 1:
                                            z_up_in = input_size[2] - inner_input_size[2]
                                            z_down_in = input_size[2]
                                        putbininer[:, x_left_in:x_right_in,y_up_in:y_down_in,z_up_in:z_down_in ] += batchmask[loc]
                                        countbininer[x_left_in:x_right_in,y_up_in:y_down_in,z_up_in:z_down_in ] += 1
                                        loc += 1
                            samplemask.append(putbininer)
                            countbin.append(countbininer)


                samplemask = torch.stack(samplemask)
                countbin = torch.stack(countbin)
                loc = 0
                for x in range(num_x):
                    for y in range(num_y):
                        for z in range(num_z):
                            x_left = x * strides[0]
                            x_right = x * strides[0] + input_size[0]
                            y_up = y * strides[1]
                            y_down = y * strides[1] + input_size[1]
                            z_up = z * strides[2]
                            z_down = z * strides[2] + input_size[2]
                            if x == num_x - 1:
                                x_left = imgx.shape[0] - input_size[0]
                                x_right = imgx.shape[0]
                            if y == num_y - 1:
                                y_up = imgx.shape[1] - input_size[1]
                                y_down = imgx.shape[1]
                            if z == num_z - 1:
                                z_up = imgx.shape[2] - input_size[2]
                                z_down = imgx.shape[2]
                            putbin[:,x_left:x_right, y_up:y_down, z_up:z_down] += samplemask[loc]
                            countbinout[x_left:x_right, y_up:y_down, z_up:z_down] += countbin[loc]
                            loc+=1
                putbin = putbin / countbinout
                fivemask.append(putbin)
            fivemask = torch.mean(torch.stack(fivemask), dim=0)
            putbin = torch.argmax(fivemask, dim=0).numpy()
            newbin = putbin[padlist[0][0] : putbin.shape[0] - padlist[0][1], padlist[1][0] : putbin.shape[1] - padlist[1][1]]

            kernewstandermask = np.zeros([resizex[inde], resizey[inde], resizez[inde]])
            kernewstandermask[stridemin[inde]: stridemax[inde]] = newbin
            kernewstandermask = resize(kernewstandermask, [orix[inde], oriy[inde], oriz[inde]], order=0, mode='constant', clip=False, preserve_range=True, anti_aliasing=False)

            ori_imgx = nib.load(os.path.join(config.oridatapath, name))
            affine = ori_imgx.affine.copy()
            header = ori_imgx.header.copy()
            nib.save(nib.Nifti1Image(kernewstandermask.astype(np.uint8), affine, header),
                     os.path.join(savepath, name.replace('Case', 'seg_case')))











