# import pandas as pd
import numpy as np
import nibabel as nib
import math
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')#
from segmentation_models_pytorch.unet import Unet
from segmentation_models_pytorch.deeplabv3 import DeepLabV3Plus
import sndhdr as snd
import torch.nn as nn
import pickle as pk
from augmentations.transforms import Compose as Compose3D
from augmentations.transforms import CropNonEmptyMaskIfExists, PadIfNeeded, PadUpAndDown
import SimpleITK as sitk
from skimage.transform import resize
import pandas as pd
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


def load_GPUS(model,file):

    state_dict= torch.load('/public/huangmeiyan/HDenseUet-master/weight/ex0/'+ file +'/M0.7275000559446765.pkl',map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
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
    def __init__(self, config, is_test=False):
        super(DualNetwork, self).__init__()
        self.config = config
        self.branch1 = DeepLabV3Plus_(encoder_name='se_resnext50_32x4d', classes=20)
        self.branch2 = DeepLabV3Plus_(encoder_name='se_resnext50_32x4d', classes=20)
        if not is_test:
            self.run_init_weight()

    def forward(self, data, step=1):
        if not self.training:
            if step==1:
                pred1 = self.branch1(data)
                return pred1[1]
            elif step == 2:
                pred1 = self.branch2(data)
                return pred1[1]
        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)


    def run_init_weight(self):
        self.init_weight(self.branch1.decoder, nn.init.kaiming_normal_,
                    nn.BatchNorm2d, self.config.bn_eps, self.config.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        self.init_weight(self.branch2.decoder, nn.init.kaiming_normal_,
                    nn.BatchNorm2d, self.config.bn_eps, self.config.bn_momentum,
                    mode='fan_in', nonlinearity='relu')

        self.init_weight(self.branch1.segmentation_head, nn.init.kaiming_normal_,
                    nn.BatchNorm2d, self.config.bn_eps, self.config.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        self.init_weight(self.branch2.segmentation_head, nn.init.kaiming_normal_,
                    nn.BatchNorm2d, self.config.bn_eps, self.config.bn_momentum,
                    mode='fan_in', nonlinearity='relu')

    def __init_weight(self, feature, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs):
        for name, m in feature.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                conv_init(m.weight, **kwargs)
            elif isinstance(m, norm_layer):
                m.eps = bn_eps
                m.momentum = bn_momentum
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_weight(self, module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                    **kwargs):
        if isinstance(module_list, list):
            for feature in module_list:
                self.__init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                              **kwargs)
        else:
            self.__init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='the number of fold for test.')
    parser.add_argument('--branch', type=str, default='2', help='use best branch model, defaut the 2 is used')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--ex', type=str, default='ex0', help='ex id for evaluate')
    parser.add_argument('--bs', type=int, default=100, help='batch size for test')
    parser.add_argument('--mainpath', type=str, default='./dataset/process2Ddata/', help='the input data with processing ')
    parser.add_argument('--infomation', type=str, default='info.csv', help='the file name of imformation that generating in processing')
    parser.add_argument('--standerpath', type=str, default='/dataset/Mask', help='the original mask without processing for inference')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # use two gpus
    div = ['sub0', 'sub1', 'sub2', 'sub3', 'sub4']
    ex = args.ex
    fold = args.fold
    sub = 'sub' + str(fold)
    bs = args.bs
    mainpath = args.mainpath
    standerpath = args.standerpath
    savepath = './log/' + ex + '/' + sub ## the output will save in the log file of every fold

    os.makedirs(savepath, exist_ok=True)
    with open('splitdataset.pkl', 'rb') as f:
        dataset = pk.load(f)
    filename = dataset[fold][1]

    df = pd.read_csv(os.path.join(mainpath, args.infomation))  ## the info.csv has been generate by process_data.py
    allname = df['name'].tolist()
    stridemin = df['stridemin'].values
    stridemax = df['stridemax'].values
    for l,file in enumerate(div):
      if l !=fold:
          continue
      else:
        # aux_params = {'classes': 62}
        model = DualNetwork(config=None, is_test=True)
        model = nn.DataParallel(model, device_ids=[0])
        modelfile = os.listdir('./weight/' + ex + '/' + sub+'/branch' + args.branch +'/')[0]
        model.load_state_dict(torch.load(os.path.join('./weight/' + ex + '/' + sub  + '/branch' + args.branch +'/', modelfile),map_location='cpu'))
      #   load_GPUS(model, file)
        model = model.cuda()
        process = Compose3D([PadUpAndDown(axis=-1)
                         ], p=1.0)
        with torch.no_grad():
            model.eval()

            for i,name in enumerate(filename):
                inde = allname.index(name)
                print(name)
                standermask = sitk.ReadImage(os.path.join(standerpath, name.replace('Case', 'mask_case')))
                space = standermask.GetSpacing()
                standermask = nib.load(os.path.join(standerpath, name.replace('Case', 'mask_case')))
                standermask = standermask.get_fdata()

                newresolutionxy = 0.34482759 * 2
                newresolutionz = 4.4000001
                rsize = [
                         round(standermask.shape[0] * space[1] / newresolutionxy),
                         round(standermask.shape[1] * space[0] / newresolutionxy),
                         round(standermask.shape[2] * space[2] / newresolutionz),
                ]
                newstandermask = resize(standermask, rsize, order=0, mode='constant',clip=False, preserve_range=True)

                imgx = nib.load(os.path.join(os.path.join(mainpath, 'image'), name))
                affine = imgx.affine.copy()
                hdr = imgx.header.copy()
                imgx =imgx.get_fdata()

                imgx = padupdown(imgx, axis=-1)
                imgx, padlist = pad(imgx, new_shape=[256, 512])
                putbin1 = np.zeros((20, imgx.shape[0], imgx.shape[1],imgx.shape[2]))
                putbin2 = np.zeros((20, imgx.shape[0], imgx.shape[1], imgx.shape[2]))
                countbin = np.zeros_like(imgx)
                input_size = [256,512,3]
                strides = [20, 20, 1]
                num_x, num_y, num_z  = gettimes(imgx.shape, input_size, strides)
                count = 0
                batch = []
                samplemask1 = []
                samplemask2 = []
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
                            inpt = imgx[x_left:x_right, y_up:y_down, z_up:z_down]
                            count += 1
                            batch.append(inpt)
                            if count % bs == 0 or count == num_x * num_y * num_z:

    #                             # CT_one = np.copy(CT_change_spacing_padding[x_left:x_right, y_up:y_down, z_top:z_botton])
                                batch = np.asarray(batch)
                                batch = torch.from_numpy(batch).float().cuda()
                                batch = batch.permute([0, 3, 1, 2])
                                temppred1 = model(batch, 1)
                                temppred1 = torch.softmax(temppred1, dim=1).cpu().numpy()
                                torch.cuda.empty_cache()
                                temppred2 = model(batch, 2)
                                temppred2 = torch.softmax(temppred2, dim=1).cpu().numpy()
                                torch.cuda.empty_cache()
                                samplemask1.append(temppred1)
                                samplemask2.append(temppred2)
                                batch = []
                samplemask1 = np.concatenate(samplemask1, axis=0)
                samplemask2 = np.concatenate(samplemask2, axis=0)
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
                            putbin1[:, x_left:x_right, y_up:y_down, z_up+1] += samplemask1[loc]
                            putbin2[:, x_left:x_right, y_up:y_down, z_up + 1] += samplemask2[loc]
                            countbin[x_left:x_right, y_up:y_down, z_up+1] += 1
                            loc+=1
                countbin = np.where(countbin == 0, 1 , countbin)
                putbin1 = putbin1 / countbin
                putbin2 = putbin2 / countbin
                putbin = (putbin1 + putbin2) / 2
                putbin = np.argmax(putbin, axis=0)
                newbin = putbin[padlist[0][0] : putbin.shape[0] - padlist[0][1], padlist[1][0] : putbin.shape[1] - padlist[1][1], 1: -1]

                kernewstandermask = np.zeros_like(newstandermask)
                kernewstandermask[stridemin[inde]: stridemax[inde]] = newbin
                nib.save(nib.Nifti1Image(kernewstandermask.astype(np.uint8), affine, hdr),os.path.join(savepath, name))










