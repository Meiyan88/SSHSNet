from segmentation_models_pytorch.deeplabv3 import DeepLabV3Plus
from nnunet.training.loss_functions import dice_loss, crossentropy
from torch.utils.data import Dataset,DataLoader
import torch
from torch.nn.functional import one_hot
import nibabel as nib
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import *
from albumentations import (HorizontalFlip,Resize, IAAPerspective, ShiftScaleRotate, CLAHE, Rotate,
Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
                            MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine, IAASharpen, IAAEmboss, Flip, OneOf, Compose,ElasticTransform, Normalize)
from medpy.metric import dc
from augmentations.transforms import Compose as Compose3D
from augmentations.transforms import CropNonEmptyMaskIfExists, PadIfNeeded, PadUpAndDown, RandomCrop
from seim_supervise_utils import mask_gen, custom_collate
import pickle as pk

class CaptchaCreator(torch.utils.data.Sampler):
    def __init__(self, count=200, usecount=250, bs=2):
        self.count = count
        self.usecount = usecount
        self.bs = bs
    # @staticmethod
    def random_seq(self):
        return [random.choice(list(range(self.count))) for _ in range(self.usecount * self.bs)]

    def shuffle(self):
        digits = self.random_seq()
        random.shuffle(digits)
        return digits

    def __iter__(self):
        return iter(self.shuffle())
    def __len__(self):
        return len(self.shuffle())


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def plot(savepath,name,x,y1,y2):
    plt.title(name)
    plt.plot(x,y1,color='red', label='train')
    plt.plot(x, y2, color='green', label='valid')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(savepath, name + '.png'))
    plt.close()

def savemodel(path,model,nowloss,bestloss):
    os.makedirs(path, exist_ok=True)
    if nowloss > bestloss:
        bestloss = nowloss
        file = os.listdir(path)

        if len(file) !=0:
            back = os.path.splitext(file[0])[1]
            if '.pkl' in back:
                os.remove(os.path.join(path, file[0]))
        torch.save(model.state_dict(),os.path.join(path, 'M'+str(nowloss) +'.pkl'))
    return bestloss

def dice1(inputs, GT):
    with torch.no_grad():
        N = GT.size(0)
        smooth = 0.2
        inputs[inputs > 0.5] = 1
        inputs[inputs <= 0.5] = 0
        GT[GT > 0.5] = 1
        GT[GT <= 0.5] = 0
        input_flat = inputs.view(N, -1)
        target_flat = GT.view(N, -1)
        intersection = input_flat * target_flat
        dice_sco = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_sco = dice_sco.sum() / N
    return dice_sco

def minmax(img):
    data = img
    imin = np.percentile(data,0.1)
    imax = np.percentile(data,99.9)
    data = ((np.clip(data,imin,imax) - imin)/ (imax - imin))
    return data


def normalize(img):
    ind = np.where(img !=0)
    mask = np.zeros(img.shape, dtype=np.int)
    mask[ind] = 1
    mean = img[ind].mean()
    std = img[ind].std()
    denominator = np.reciprocal(std)
    img = (img - mean) * denominator
    return mask * img

class Data(Dataset):
    def __init__(self, root, name, havelabel=True, istrain=True):
        self.root = root
        self.name = name
        self.havelabel=havelabel
        if self.havelabel:
            self.process = Compose3D([PadUpAndDown(axis=-1),
                                      PadIfNeeded(shape=[256, 512], always_apply=True),
                                      CropNonEmptyMaskIfExists(shape=[256, 512, 3], always_apply=True)],
                                     p=1.0)
        else:
            self.process = Compose3D([PadUpAndDown(axis=-1),
                                      PadIfNeeded(shape=[256, 512], always_apply=True),
                                      RandomCrop(shape=[256, 512, 3], always_apply=True)],
                                     p=1.0)
        self.istrain = istrain
        if self.istrain:
            self.compose = Compose([#Rotate(p=0.3),
                                    Flip(p=0.3),

                                    GaussNoise(var_limit=(0.0, 3.0),p=0.4),
                                        # IAAAdditiveGaussianNoise(),

                                    OneOf([
                                        MotionBlur(),
                                        Blur(blur_limit=3, ),
                                    ], p=0.3),
                                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
                                    OneOf([
                                        OpticalDistortion(),
                                        GridDistortion(),
                                        # IAAPiecewiseAffine(),
                                    ], p=0.3),
                                    OneOf([
                                        IAASharpen(),
                                        IAAEmboss(),
                                    ], p=0.3),
                                    # CropNonEmptyMaskIfExists(height=100, width=100, p=0.3),
                                    # PadIfNeeded(min_height=192,min_width=192,border_mode=cv2.BORDER_CONSTANT,value=0)
                                    ], p=0.5)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, item):

        name = self.name[item]
        data = self.getdata(name)
        return  data

    def getdata(self, name):
        if self.havelabel:
            img = nib.load(os.path.join(os.path.join(self.root, 'image'), name))
            img = img.get_fdata()
            mask = nib.load(os.path.join(os.path.join(self.root, 'mask'), name))
            mask = mask.get_fdata()

        # print(np.max(mask))
            data = {'image': img, 'mask': mask}
            data = self.process(**data)
            img = data['image']
            mask = data['mask'][:,:,1]
            if self.istrain:
                data = {'image': img, 'mask': mask}
                data = self.compose(**data)
                img = data['image']
                mask = data['mask']
            # else:
            #     x = np.asarray(x).astype(np.float32)
            img = np.transpose(img, axes=[2, 0, 1])
            img = torch.from_numpy(img).type(torch.float32)
            mask = torch.from_numpy(mask).long()
            return {'data':img, 'label':mask}
        else:
            img = nib.load(os.path.join(os.path.join(self.root, 'image'), name))
            img = img.get_fdata()
            data = {'image': img}
            data = self.process(**data)
            img = data['image']
            if self.istrain:
                data = {'image': img}
                data = self.compose(**data)
                img = data['image']
            img = np.transpose(img, axes=[2, 0, 1])
            img = torch.from_numpy(img).type(torch.float32)
            return {'data':img}





    def normalize(self, x):
        ma = np.max(np.max(x, axis=0), axis=0)
        mi = np.min(np.min(x, axis=0), axis=0)
        return  (x - mi) * 255.0 / (ma - mi)

def hiddenloss(own,y):
    newy = y.detach().cpu().numpy()
    losssame = 0
    for i in range(y.size(0)):
        ind = np.where(newy == newy[i])[0]
        for j in range(ind.shape[0]):
            losssame += criterion(own[i],own[ind[j]])

    return losssame


def progress(istrain, epoch, model,optimizer, loader,writer):
    index = Index()
    total_loss1 = AverageMeter()
    total_loss2 = AverageMeter()
    Mean_dice1 = AverageMeter()
    Mean_dice2 = AverageMeter()


    # labels = np.empty(len_y)
    # predicts = np.empty(len_y)
    # score = np.empty(len_y)
    for i, data in enumerate(loader):
        # print(index(i,len(loader)-1), end='')
        x = data['data']
        y = data['label']
        x = x.cuda().float()
        y = y.cuda().long()
        # print(torch.max(x))
        # print(torch.min(x))
        if istrain:
            model.train()
        else:
            model.eval()

        output1 = model(x, 1)
        output2 = model(x, 2)

        Seg_one_hot = (one_hot(y, 20)).permute(0,3,1,2)

        loss1 = criteion_ce_dc(output1, torch.unsqueeze(y, 1))
        predict =torch.softmax(output1,dim=1)
        Seg_prediction = torch.argmax(predict.cpu(), dim=1, keepdim=True).long()
        Seg_prediction_one_hot = torch.zeros((Seg_prediction.size(0), 20, Seg_prediction.size(2),
                                              Seg_prediction.size(3))).scatter_(1, Seg_prediction, 1)
        alldc1 = []
        for j in range(20):
            alldc1.append(dc(Seg_one_hot[:, j,:,:].cpu().detach().numpy(), Seg_prediction_one_hot[:, j, :, :].cpu().detach().numpy()))
        meandc1 = np.mean(np.asarray(alldc1))

        loss2 = criteion_ce_dc(output2, torch.unsqueeze(y, 1))
        predict =torch.softmax(output2,dim=1)
        Seg_prediction = torch.argmax(predict.cpu(), dim=1, keepdim=True).long()
        Seg_prediction_one_hot = torch.zeros((Seg_prediction.size(0), 20, Seg_prediction.size(2),
                                              Seg_prediction.size(3))).scatter_(1, Seg_prediction, 1)
        alldc2 = []
        for j in range(20):
            alldc2.append(dc(Seg_one_hot[:, j,:,:].cpu().detach().numpy(), Seg_prediction_one_hot[:, j, :, :].cpu().detach().numpy()))
        meandc2 = np.mean(np.asarray(alldc2))

        # if y.size(0) == bs:
        #     start = i*bs
        #     endt = (i+1)*bs
        # else:
        #     start = i*bs
        #     endt = len_y
        # labels[start:endt] = y.cpu().numpy()
        # predicts[start:endt] = predict.cpu().numpy()
        # score[start:endt] = pred.detach().cpu().numpy()
        # if istrain:
        #     optimizer.zero_grad()
        #     loss.backward()
        #     nn.utils.clip_grad_norm_(model.parameters(), 12)
        #     # for i, (name, parms) in enumerate(model.named_parameters()):
        #     #     if i == 0:
        #     #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        #     #               ' -->grad_value:', parms.grad)
        #     #         break
        #     optimizer.step()
        total_loss1.update(loss1.item()*y.size(0), y.size(0))
        Mean_dice1.update(meandc1*y.size(0), y.size(0))

        total_loss2.update(loss2.item() * y.size(0), y.size(0))
        Mean_dice2.update(meandc2 * y.size(0), y.size(0))

        # if not istrain:
        #     if epoch % 5 == 0  and i == 1:
        #         pred = make_grid(Seg_prediction, test_batch_size)
        #         mask = make_grid(torch.unsqueeze(y.cpu(), dim=1), test_batch_size)
        #         writer.add_image('pred', pred, epoch)
        #         writer.add_image('mask', mask, epoch)

    writer.add_scalar('loss1', total_loss1.avg, epoch)
    writer.add_scalar('MeanDice1', Mean_dice1.avg, epoch)

    writer.add_scalar('loss2', total_loss2.avg, epoch)
    writer.add_scalar('MeanDice2', Mean_dice2.avg, epoch)


    print('epoch:{0} validloss:{1}, validloss:{2}, Meandice:{3}, Meandice:{4}'.format(epoch, total_loss1.avg, total_loss2.avg,
                                                                                      Mean_dice1.avg, Mean_dice2.avg))
    return Mean_dice1.avg, Mean_dice2.avg


def trainer(writer):

    total_loss1 = AverageMeter()
    total_loss2 = AverageMeter()
    total_loss3 = AverageMeter()
    total_loss = AverageMeter()
    Mean_dice1 = AverageMeter()
    Mean_dice2 = AverageMeter()
    model.train()
    dataloader = iter(trainloader)
    unsupervised_dataloader_0 = iter(unsuper0loader)
    unsupervised_dataloader_1 = iter(unsuper0loader)


    ''' supervised part '''
    for idx in range(len(trainloader)):
        optimizer_l.zero_grad()
        optimizer_r.zero_grad()

        minibatch = dataloader.next()
        unsup_minibatch_0 = unsupervised_dataloader_0.next()
        unsup_minibatch_1 = unsupervised_dataloader_1.next()

        imgs = minibatch['data']
        gts = minibatch['label']
        unsup_imgs_0 = unsup_minibatch_0['data']
        unsup_imgs_1 = unsup_minibatch_1['data']
        mask_params = unsup_minibatch_0['mask_params']

        imgs = imgs.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)
        unsup_imgs_0 = unsup_imgs_0.cuda(non_blocking=True)
        unsup_imgs_1 = unsup_imgs_1.cuda(non_blocking=True)
        mask_params = mask_params.cuda(non_blocking=True)

        # unsupervised loss on model/branch#1
        batch_mix_masks = mask_params
        unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks

        # plt.subplot(141)
        # plt.imshow(unsup_imgs_0.detach().cpu().numpy()[0, 1])
        # plt.subplot(142)
        # plt.imshow(unsup_imgs_1.detach().cpu().numpy()[0, 1])
        # plt.subplot(143)
        # plt.imshow(unsup_imgs_mixed.detach().cpu().numpy()[0, 1])
        # plt.subplot(144)
        # plt.imshow(mask_params.detach().cpu().numpy()[0, 0])
        # plt.show()

        with torch.no_grad():
            # Estimate the pseudo-label with branch#1 & supervise branch#2
            _, logits_u0_tea_1 = model(unsup_imgs_0, step=1)
            _, logits_u1_tea_1 = model(unsup_imgs_1, step=1)
            logits_u0_tea_1 = logits_u0_tea_1.detach()
            logits_u1_tea_1 = logits_u1_tea_1.detach()
            # Estimate the pseudo-label with branch#2 & supervise branch#1
            _, logits_u0_tea_2 = model(unsup_imgs_0, step=2)
            _, logits_u1_tea_2 = model(unsup_imgs_1, step=2)
            logits_u0_tea_2 = logits_u0_tea_2.detach()
            logits_u1_tea_2 = logits_u1_tea_2.detach()

        # Mix teacher predictions using same mask
        # It makes no difference whether we do this with logits or probabilities as
        # the mask pixels are either 1 or 0
        logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
        _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
        ps_label_1 = ps_label_1.long()
        logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
        _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
        ps_label_2 = ps_label_2.long()

        # Get student#1 prediction for mixed image
        _, logits_cons_stu_1 = model(unsup_imgs_mixed, step=1)
        # Get student#2 prediction for mixed image
        _, logits_cons_stu_2 = model(unsup_imgs_mixed, step=2)

        cps_loss = criterion_ce(logits_cons_stu_1.cuda(1, non_blocking=True), ps_label_2.unsqueeze(1).cuda(1, non_blocking=True)
                                  ) + criterion_ce(logits_cons_stu_2.cuda(1, non_blocking=True), ps_label_1.unsqueeze(1).cuda(1, non_blocking=True))
        # dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
        # cps_loss = cps_loss / engine.world_size
        # cps_loss = cps_loss * config.cps_weight

        # supervised loss on both models
        _, sup_pred_l = model(imgs, step=1)
        _, sup_pred_r = model(imgs, step=2)

        loss_sup = criteion_ce_dc(sup_pred_l.cuda(1, non_blocking=True), gts.unsqueeze(1).cuda(1, non_blocking=True))
        # dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
        # loss_sup = loss_sup / engine.world_size

        loss_sup_r = criteion_ce_dc(sup_pred_r.cuda(1, non_blocking=True), gts.unsqueeze(1).cuda(1, non_blocking=True))
        # dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
        # loss_sup_r = loss_sup_r / engine.world_size
        # current_idx = epoch * config.niters_per_epoch + idx
        # lr = lr_policy.get_lr(current_idx)

        # print(len(optimizer.param_groups))
        # optimizer_l.param_groups[0]['lr'] = lr
        # optimizer_l.param_groups[1]['lr'] = lr
        # for i in range(2, len(optimizer_l.param_groups)):
        #     optimizer_l.param_groups[i]['lr'] = lr
        # optimizer_r.param_groups[0]['lr'] = lr
        # optimizer_r.param_groups[1]['lr'] = lr
        # for i in range(2, len(optimizer_r.param_groups)):
        #     optimizer_r.param_groups[i]['lr'] = lr

        loss = loss_sup + loss_sup_r + cps_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.module.branch1.parameters(), 12)
        nn.utils.clip_grad_norm_(model.module.branch2.parameters(), 12)
        optimizer_l.step()
        optimizer_r.step()

        Seg_one_hot = (one_hot(gts.detach().cpu(), 20)).permute(0, 3, 1, 2)

        predict = torch.softmax(sup_pred_l.detach().cpu(), dim=1)
        Seg_prediction = torch.argmax(predict.cpu(), dim=1, keepdim=True).long()
        Seg_prediction_one_hot = torch.zeros((Seg_prediction.size(0), 20, Seg_prediction.size(2),
                                              Seg_prediction.size(3))).scatter_(1, Seg_prediction, 1)
        alldc1 = []
        for j in range(20):
            alldc1.append(dc(Seg_one_hot[:, j, :, :].cpu().detach().numpy(),
                             Seg_prediction_one_hot[:, j, :, :].cpu().detach().numpy()))
        meandc1 = np.mean(np.asarray(alldc1))

        predict = torch.softmax(sup_pred_r.detach().cpu(), dim=1)
        Seg_prediction = torch.argmax(predict.cpu(), dim=1, keepdim=True).long()
        Seg_prediction_one_hot = torch.zeros((Seg_prediction.size(0), 20, Seg_prediction.size(2),
                                              Seg_prediction.size(3))).scatter_(1, Seg_prediction, 1)
        alldc2 = []
        for j in range(20):
            alldc2.append(dc(Seg_one_hot[:, j, :, :].cpu().detach().numpy(),
                             Seg_prediction_one_hot[:, j, :, :].cpu().detach().numpy()))
        meandc2 = np.mean(np.asarray(alldc2))

        total_loss1.update(loss_sup.item() * gts.size(0), gts.size(0))
        Mean_dice1.update(meandc1 * gts.size(0), gts.size(0))

        total_loss2.update(loss_sup_r.item() * gts.size(0), gts.size(0))
        Mean_dice2.update(meandc2 * gts.size(0), gts.size(0))

        total_loss3.update(cps_loss.item() * gts.size(0), gts.size(0))
        total_loss.update(loss.item() * gts.size(0), gts.size(0))

    writer.add_scalar('loss1', total_loss1.avg, epoch)
    writer.add_scalar('MeanDice1', Mean_dice1.avg, epoch)

    writer.add_scalar('loss2', total_loss2.avg, epoch)
    writer.add_scalar('MeanDice2', Mean_dice2.avg, epoch)

    writer.add_scalar('lossun', total_loss3.avg, epoch)
    writer.add_scalar('loss', total_loss.avg, epoch)

    print('epoch:{0} Trainloss1:{1}, Trainloss2:{2}, Meandice1:{3}, Meandice2:{4}'.format(epoch, total_loss1.avg,
                                                                                      total_loss2.avg,
                                                                                      Mean_dice1.avg, Mean_dice2.avg))

def readtxt(name):
    allname = []
    alllabel = []
    with open(name) as f:
        for row in f:
            allname.append(row.split(' ')[0])
            alllabel.append(int(row.split(' ')[1].split('\n')[0]))
    return allname,np.asarray(alllabel)

def load_GPUS(model,path,old,new):
    file = os.listdir(path)
    if len(file) != 0:
        for name in file:
            if endWith(name, '.pkl'):
                    state_dict= torch.load(os.path.join(path, name), map_location={'cuda:' + str(old): 'cuda:' + str(new)})
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def writelog(path,filename,epoch, trainloss,validloss):
    with open(os.path.join(path,filename), 'a') as f:
        f.write(str(epoch) + ' ' + str(trainloss) +' '+ str(validloss) + '\n')

def smoothVal(nowval, lastsmoothval= None):
    if lastsmoothval== None:
        return nowval
    else:
        return 0.9 * lastsmoothval + 0.1 * nowval



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
        self.branch1 = DeepLabV3Plus_(encoder_name='se_resnext50_32x4d', classes=20)
        self.branch2 = DeepLabV3Plus_(encoder_name='se_resnext50_32x4d', classes=20)
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


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int,default=0, help='number fold to train.')
    parser.add_argument('--gpuid', type=str, default='0',help='use gpu id.')
    parser.add_argument('--exid', type=str, default='ex0', help='the number of experiment.')
    parser.add_argument('--train_batch_size', type=int, default=8, help='train_batch_size.')
    parser.add_argument('--test_batch_size', type=int, default=8, help='test_batch_size.')
    parser.add_argument('--datapath', type=str, default='./dataset/process2Ddata', help='the data path including image and mask.')
    parser.add_argument('--seed', type=int, default=100, help='random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='train epochs.')
    parser.add_argument('--bn_eps', type=float,default=1e-5,  help='BN parameter.')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='BN parameter.')
    parser.add_argument('--cutmix_mask_prop_range', type=tuple,default=(0.25, 0.5),  help='cut range for semi-supervise.')
    parser.add_argument('--cutmix_boxmask_n_boxes', type=int, default=3, help='box number .')
    parser.add_argument('--cutmix_boxmask_fixed_aspect_ratio', type=bool, default=False, help='mask rate .')
    parser.add_argument('--cutmix_boxmask_by_size', type=bool, default=False, help='box by size.')
    parser.add_argument('--cutmix_boxmask_outside_bounds', type=bool, default=False, help='box out siez bounds.')
    parser.add_argument('--cutmix_boxmask_no_invert', type=bool, default=False, help='where invert.')
    config = parser.parse_args()

    weightpath = os.path.join('./weight',  config.exid)
    logpath = os.path.join('./log',  config.exid)
    losspath = os.path.join('./loss',  config.exid)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    div = ['sub0', 'sub1', 'sub2', 'sub3', 'sub4']  ## five fold
    for i, sub in enumerate(div):
        if i != config.fold:
            continue
        else:
            torch.cuda.empty_cache()
            os.makedirs(os.path.join(weightpath, sub), exist_ok=True)
            os.makedirs(os.path.join(logpath, sub), exist_ok=True)
            os.makedirs(os.path.join(losspath, sub), exist_ok=True)
            train_writer = SummaryWriter(log_dir=os.path.join(os.path.join(losspath, sub), 'train'))
            test_writer = SummaryWriter(log_dir=os.path.join(os.path.join(losspath, sub), 'test'))
            model = DualNetwork(config).cuda()
            model = nn.DataParallel(model, device_ids=[0])
            # model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=0).cuda()


            mask_generator = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range,
                                                       n_boxes=config.cutmix_boxmask_n_boxes,
                                                       random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                                       prop_by_area=not config.cutmix_boxmask_by_size,
                                                       within_bounds=not config.cutmix_boxmask_outside_bounds,
                                                       invert=not config.cutmix_boxmask_no_invert)

            add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
                mask_generator
            )
            collate_fn = custom_collate.SegCollate()
            mask_collate_fn = custom_collate.SegCollate(batch_aug_fn=add_mask_params_to_batch)
            # model = nn.DataParallel(model, device_ids=[5,7])
            optimizer_l = torch.optim.Adam(model.module.branch1.parameters(), lr=1e-3)
            scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_l, milestones=[50, 400], last_epoch=-1)
            optimizer_r = torch.optim.Adam(model.module.branch2.parameters(), lr=1e-3)
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_r, milestones=[50, 400], last_epoch=-1)

            with open('./splitdataset.pkl', 'rb') as f:
                dataset = pk.load(f)
            trainfile = dataset[i][0]
            validfile = dataset[i][1]
            with open('./unlabeldataset.pkl', 'rb') as f:
                test1file = pk.load(f)

            trainset = Data(root=config.datapath, name=trainfile,istrain=True)
            testset = Data(root=config.datapath, name=validfile,istrain=False)
            unsuper0 = Data(root=config.datapath, name=test1file,havelabel=False, istrain=True)
            unsuper1 = Data(root=config.datapath, name=test1file, havelabel=False, istrain=True)

            trainloader = DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True, pin_memory=True,num_workers=8, drop_last=True, collate_fn=collate_fn)

            unsuper0loader = DataLoader(unsuper0, batch_size=config.train_batch_size,
                                     sampler=CaptchaCreator(count=len(test1file), usecount=len(trainloader),
                                     bs=config.train_batch_size),  shuffle=False, pin_memory=True,num_workers=4, collate_fn=mask_collate_fn)
            unsuper1loader = DataLoader(unsuper1, batch_size=config.train_batch_size,
                                        sampler=CaptchaCreator(count=len(test1file), usecount=len(trainloader),
                                                               bs=config.train_batch_size), shuffle=False, pin_memory=True,
                                        num_workers=4, collate_fn=collate_fn)
            testloader = DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, pin_memory=True,num_workers=4)
            criterion_ce = crossentropy.RobustCrossEntropyLoss()
            criteion_ce_dc = dice_loss.DC_and_CE_loss(soft_dice_kwargs={}, ce_kwargs=dict())

            trainloss = []
            trainacc = []
            testloss = []
            testacc = []
            bestdsc1 = 0
            smootheval1 = None
            bestdsc2 = 0
            smootheval2 = None
            for epoch in range(config.epochs):
                trainer(train_writer)
                scheduler1.step()
                scheduler2.step()
                with torch.no_grad():
                    dsc1,dsc2 = progress(istrain=False, epoch=epoch, model=model, optimizer=None,
                                         loader=testloader,writer=test_writer)
                    smootheval1 = smoothVal(dsc1, lastsmoothval=smootheval1)
                    smootheval2 = smoothVal(dsc2, lastsmoothval=smootheval2)
                # scheduler.step()
                bestdsc1 = savemodel(os.path.join(os.path.join(weightpath, sub), 'branch1'),model,smootheval1,bestdsc1)
                bestdsc2 = savemodel(os.path.join(os.path.join(weightpath, sub), 'branch2'), model, smootheval2,
                                     bestdsc2)
