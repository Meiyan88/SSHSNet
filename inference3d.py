# import pandas as pd
import numpy as np
import nibabel as nib
import math
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('AGG')#
from segmentation_models_pytorch.unet import Unet
from segmentation_models_pytorch.deeplabv3 import DeepLabV3Plus
# from remodel import DeepLabV3PlusPix
import sndhdr as snd
import torch.nn as nn
import pickle as pk
from augmentations.transforms import Compose as Compose3D
from augmentations.transforms import CropNonEmptyMaskIfExists, PadIfNeeded, PadUpAndDown
from data_parallel_my_v2 import BalancedDataParallel
from skimage.transform import resize
from seresnet import se_resnext50, DeepLabV3PlusDecoder, se_resnet50
import torch.nn.functional as F
import SimpleITK as sitk
import time
from spatial_attention_module import DualCrossAttModule,TrippleDualCrossAttModule

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
    print(file)
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
    def __init__(self,  encoder_name='se_resnext50_32x4d', classes=20,
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
        self.branch1 = DeepLabV3Plus_(encoder_name='se_resnext50_32x4d', classes=20)
        self.branch2 = DeepLabV3Plus_(encoder_name='se_resnext50_32x4d', classes=20)


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

        # img = f.cpu().numpy()
        # p1 = torch.argmax(m, dim=0).cpu().numpy()
        #
        # plt.subplot(121)
        # plt.imshow(img[0, 1,:,:])
        # plt.subplot(122)
        # plt.imshow(p1[:, :, 0])
        # # plt.subplot(133)
        # # plt.imshow(img[:, :, 6])
        # plt.show()

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

class DeeplabV3Plus3D_cadm(nn.Module):

    def __init__(self,  num_classes, encoder='se_resnetx50'):
        super(DeeplabV3Plus3D_cadm, self).__init__()
        if encoder == 'se_resnet50':
            self.encoder = se_resnet50()
        elif encoder == 'se_resnetx50':
            self.encoder = se_resnext50()
        self.assp = DeepLabV3PlusDecoder(encoder_channels=[64, 256, 512, 1024, 2048])

        self.conv1 = nn.Conv3d(512, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(256)
        self.relu = nn.ReLU(inplace=True)

        self.segmentation = nn.Conv3d(256, num_classes, kernel_size=1)
        # self.dual_att_module_input = DualCrossAttModule(gate_channels=21, no_channel=True)
        self.dual_att_module_feature = DualCrossAttModule(gate_channels=256, no_channel=True)

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



class TTA(object):
    def __init__(self, flip):
        self.Flip = flip

    def __call__(self, x, original_last=True, reverse=False):
        if reverse:
            img = []
            if original_last:
                for i in range(len(x) - 1):
                    img.append(torch.flip(x[i], [self.Flip[i] + 1]))
                img.append(x[-1])
            else:
                for i in range(1, len(x), 1):
                    img.append(torch.flip(x[i], [self.Flip[i-1] + 1]))
                img.append(x[0])
        else:
            img = []
            for i in range(len(self.Flip)):
                img.append(np.flip(x, self.Flip[i]))
        return img




if __name__ == '__main__':
    # path = '/public/huangmeiyan/validdata/predictby2d/'
    # div = ['sub0', 'sub1', 'sub2', 'sub3', 'sub4']
    # file = os.listdir(os.path.join(path, 'sub0'))
    # savepath = '/public/huangmeiyan/validdata/predictby2d/concate'
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    # for name in file:
    #     t = 0
    #     for sub in div:
    #         x = nib.load(os.path.join(os.path.join(path, sub), name))
    #         t += x.get_fdata()
    #         if sub == 'sub0':
    #             affine = x.affine.copy()
    #             hdr = x.header.copy()
    #     t /= 5
    #     t = np.where(t > 0.5, 2, 0)
    #     nib.save(nib.Nifti1Image(t,affine,hdr), os.path.join(savepath, name))



    # df = pd.read_csv('/public/huangmeiyan/segresult/ex18_post/Stats_Validation_final.csv')
    # wt = df['Dice_WT'][0:66]
    # name = df['Label'][0:66]
    # ind = np.where(np.array(wt)< 0.85)[0]
    # use = name[ind].tolist()
    # filepath = '/public/huangmeiyan/validdata/MICCAI_BraTS_2018_Data_Validation_N4/'
    # path = '/public/huangmeiyan/validdata/predthredori_post_ex18/'
    # times = 10
    # mode = ['t1', 't1ce', 't2', 'flair']
    # pad = PadIfNeeded([192,192,160])
    # model = denseunet_3d(1, input_size=192, input_cols=160)
    # model.load_weights('/public/huangmeiyan/3DWT/weight/ex18/denseunet_brain_segment.h5', by_name=True)
    # for i,name in enumerate(use):
    #     mask = nib.load(os.path.join(path,name +'.nii.gz'))
    #     mask = mask.get_fdata()
    #     img = np.where(mask > 0, 1, 0).astype(np.uint8)
    #     for j in range(times):
    #         img = binary_dilation(img)
    #     connect_regions = label(img, connectivity=1, background=0)
    #     props = regionprops(connect_regions)
    #     ix,iy,iz,mx,my,mz = props[0].bbox
    #     name = name.replace('BraTS19','Brats18')
    #     x = nib.load(os.path.join(os.path.join(filepath, name), name + '_t1.nii.gz'))
    #     x = np.expand_dims(normalize(x.get_fdata()), axis=-1)
    #     x1 = nib.load(os.path.join(os.path.join(filepath, name), name + '_t1ce.nii.gz'))
    #     x1 = np.expand_dims(normalize(x1.get_fdata()), axis=-1)
    #     x2 = nib.load(os.path.join(os.path.join(filepath, name), name + '_t2.nii.gz'))
    #     x2 = np.expand_dims(normalize(x2.get_fdata()), axis=-1)
    #     x3 = nib.load(os.path.join(os.path.join(filepath, name), name + '_flair.nii.gz'))
    #     x3 = np.expand_dims(normalize(x3.get_fdata()), axis=-1)
    #     x = np.concatenate([x, x1, x2, x3], axis=-1)
    #     new = x[ix:mx, iy:my, iz:mz,:]
    #     new = pad.apply(new)
    #     x = np.expand_dims(new, axis=0)
    #     pred = np.squeeze(model.predict_on_batch(x))
    #
    # path = '/public/huangmeiyan/train_n4/'
    # filname = os.listdir(path)
    # savepath = '/public/huangmeiyan/center_mask_128_128_64/'
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    # times = 5
    # mode = ['t1', 't1ce', 't2', 'flair']
    # for i,name in enumerate(filname):
    #   # if name == 'Brats18_TCIA03_338_1':
    #     print(name)
    #     if not os.path.exists(os.path.join(savepath, name)):
    #         os.makedirs(os.path.join(savepath,name))
    #     x = nib.load(os.path.join(os.path.join(path,name),'seg.nii.gz'))
    #     affine = x.affine.copy()
    #     hdr = x.header.copy()
    #     mask = x.get_fdata()
    #     img = np.where(mask > 0, 1, 0).astype(np.uint8)
    #     for j in range(times):
    #         img = binary_dilation(img)
    #     connect_regions = label(img, connectivity=1, background=0)
    #     props = regionprops(connect_regions)
    #     for n in range(len(props)):
    #         if n == 0:
    #             ix, iy, iz, mx, my, mz = props[n].bbox
    #         else:
    #             nix, niy, niz, nmx, nmy, nmz = props[n].bbox
    #             ix,mx = cmp(ix,mx, nix,nmx)
    #             iy, my = cmp(iy, my, niy, nmy)
    #             iz, mz = cmp(iz, mz, niz, nmz)
    #
    #     ix,mx = fix(ix,mx,length=128,maxx=240)
    #     iy,my = fix(iy,my,length=128,maxx=240)
    #     iz,mz = fix(iz,mz,length=64,maxx=155)
    #     # print((mx - ix), (my- iy), (mz - iz))
    #     mask = mask[ix:mx, iy:my, iz:mz]
    #     nib.save(nib.Nifti1Image(mask,affine,hdr), os.path.join(os.path.join(savepath,name),'seg.nii.gz'))
    #     for j,modename in enumerate(mode):
    #         x = nib.load(os.path.join(os.path.join(path, name), modename + '.nii.gz'))
    #         x = normalize(x.get_fdata())
    #         x = x[ix:mx, iy:my, iz:mz]
    #         nib.save(nib.Nifti1Image(x, affine, hdr), os.path.join(os.path.join(savepath, name), modename + '.nii.gz'))

    # for i,name in enumerate(filename):
    #     x = nib.load(os.path.join(os.path.join(path, name),  'seg.nii.gz'))
    #     x = x.get_fdata()
    #     print(x.shape, name)
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='Random seed.')
    parser.add_argument('--gpu', type=str,default="0, 1", help='gpu id ')
    parser.add_argument('--branch_best', type=str, default="branch2", help='best_branch.')
    parser.add_argument('--ex', type=str, default="ex65", help='best_branch.')
    parser.add_argument('--bs', type=int, default=25, help='batch size for test')
    parser.add_argument('--mainpath', type=str, default='./dataset/process3Ddata/', help='the input data with processing ')
    parser.add_argument('--infomation', type=str, default='info.csv', help='the file name of imformation that generating in processing')
    parser.add_argument('--standerpath', type=str, default='./train/Mask', help='the original mask without processing for inference')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # use two gpus
    div = ['sub0', 'sub1', 'sub2', 'sub3', 'sub4']
    ex = args.ex
    fold = args.fold
    sub = 'sub' + str(fold)
    bs = args.bs
    mainpath = args.mainpath
    standerpath = args.standerpath
    savepath = './log/' + ex + '/' + sub
    df = pd.read_csv(os.path.join(mainpath, args.infomation))
    allname = df['name'].tolist()
    stridemin = df['stridemin'].values
    stridemax = df['stridemax'].values


    if not os.path.exists(savepath):
        os.makedirs(savepath)
    with open('splitdataset.pkl', 'rb') as f:
        dataset = pk.load(f)
    filename = dataset[fold][1]

    for l,file in enumerate(div):
      if l  != fold:
          continue
      else:
            model2d = DualNetwork(None).cuda(1)
            model2d = load_GPUS(model2d,
                              os.path.join(os.path.join(os.path.join('./weight', 'ex49'), sub), args.branch_best))

            model3d = DeeplabV3Plus3D_tripple(num_classes=20,
                                    encoder='se_resnetx50', mode='cat').cuda()
            model3d = BalancedDataParallel(15, model3d, dim=0,device_ids=[0,1]).cuda()
            modelfile = os.listdir('./weight/' + ex + '/' + sub)[0]
            model3d.load_state_dict(torch.load(os.path.join('./weight/' + ex + '/' + sub , modelfile),map_location='cpu'))
            with torch.no_grad():
                model3d.eval()
                model2d.eval()
                # if not os.path.exists(os.path.join(savepath,file)):
                #     os.makedirs(os.path.join(savepath,file))
                for i,name in enumerate(filename):
                    inde = allname.index(name)
                    # print(name)
                    # start = time.time()
                    standermask = sitk.ReadImage(os.path.join(standerpath, name.replace('Case', 'mask_case')))
                    space = standermask.GetSpacing()
                    standermask = nib.load(os.path.join(standerpath, name.replace('Case', 'mask_case')))
                    standermask = standermask.get_fdata()

                    newresolutionxy = 0.34482759
                    newresolutionz = 4.4000001
                    rsize = [
                        round(standermask.shape[0] * space[1] / newresolutionxy),
                        round(standermask.shape[1] * space[0] / newresolutionxy),
                        round(standermask.shape[2] * space[2] / newresolutionz),
                    ]
                    newstandermask = resize(standermask, rsize, order=0, mode='constant', clip=False,
                                            preserve_range=True)



                    imgx = nib.load(os.path.join(os.path.join(mainpath, 'image'), name))
                    affine = imgx.affine.copy()
                    hdr = imgx.header.copy()
                    imgx =imgx.get_fdata()

                    # imgx = padupdown(imgx, axis=-1)
                    imgx, padlist = pad(imgx, new_shape=[512, 1024])
                        # nix, niy, niz, nmx, nmy, nmz = ix, iy, iz, mx, my, mz
                        # name = name.replace('BraTS19','Brats18').split('.')[0]
                    input_size = [512,1024,12]

                    strides = [100, 100, 6]


                    inner_input_size = [192, 192, 12]
                    inner_strides = [81, 81, 1]
                    num_x, num_y, num_z = gettimes(imgx.shape, input_size, strides)
                    num_x_in, num_y_in, num_z_in = gettimes(input_size, inner_input_size, inner_strides)

                    putbin = torch.zeros((20, imgx.shape[0], imgx.shape[1], imgx.shape[2]))
                    countbinout = torch.zeros([imgx.shape[0], imgx.shape[1], imgx.shape[2]])

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
                                img_ = np.transpose(img_, axes=[1, 2, 3, 0])
                                img_ = resize(img_, output_shape=[256, 512, 12], order=1, mode='constant', clip=False, preserve_range=True)
                                img_ = np.transpose(img_, axes=[2, 3, 0, 1])

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
                                            if count % bs == 0 or count == num_x_in * num_y_in * num_z_in:
                                                batch = np.expand_dims(np.asarray(batch), axis=1)
                                                batch = torch.from_numpy(batch).float().cuda()
                                                iner2d = torch.from_numpy(img_).float().cuda(1)
                                                # batch = batch.permute([0, 3, 1, 2])
                                                locfeature, locscore = model2d(iner2d, locdict)
                                                locfeature = locfeature.cuda()
                                                locscore = locscore.cuda()
                                                temppred = model3d(batch, locfeature,  locscore)
                                                temppred = torch.softmax(temppred, dim=1).cpu()
                                                batchmask.append(temppred)
                                                batch = []
                                                locdict = []
                                                torch.cuda.empty_cache()

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
                    putbin = torch.argmax(putbin, dim=0).numpy()
                    newbin = putbin[padlist[0][0] : putbin.shape[0] - padlist[0][1], padlist[1][0] : putbin.shape[1] - padlist[1][1]]

                    kernewstandermask = np.zeros_like(newstandermask)
                    # t = [stridemin[inde], stridemax[inde]]
                    kernewstandermask[stridemin[inde]: stridemax[inde]] = newbin
                    # kernewstandermask = resize(kernewstandermask, standermask.shape, order=0, mode='constant', clip=False, preserve_range=True, anti_aliasing=False)
                    nib.save(nib.Nifti1Image(kernewstandermask.astype(np.uint8), affine, hdr),
                             os.path.join(savepath, name))










