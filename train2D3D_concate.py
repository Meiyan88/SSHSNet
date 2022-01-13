from segmentation_models_pytorch.deeplabv3 import DeepLabV3Plus
from nnunet.training.loss_functions import dice_loss, crossentropy
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import *
import torch
import nibabel as nib
from skimage.transform import resize
from augmentations.transforms import Compose, GaussianNoise, Flip, \
    PadIfNeeded, CropNonEmptyMaskIfExists, RandomRotate90, RandomScale2, PadUpAndDown, RotatePseudo2D, \
    ElasticTransformPseudo2D, CropNonEmptyMaskIfExistsBalance, Resize
from seresnet import se_resnext50, DeepLabV3PlusDecoder, se_resnet50
from medpy.metric import dc
import pickle as pk
from spatial_attention_module import  TrippleDualCrossAttModule

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


def plot(savepath, name, x, y1, y2):
    plt.title(name)
    plt.plot(x, y1, color='red', label='train')
    plt.plot(x, y2, color='green', label='valid')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(savepath, name + '.png'))
    plt.close()


def savemodel(path, model, nowloss, bestloss):
    if nowloss > bestloss:
        bestloss = nowloss
        file = os.listdir(path)

        if len(file) != 0:
            back = os.path.splitext(file[0])[1]
            if '.pkl' in back:
                os.remove(os.path.join(path, file[0]))
        torch.save(model.state_dict(), os.path.join(path, 'M' + str(nowloss) + '.pkl'))
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
    imin = np.percentile(data, 0.1)
    imax = np.percentile(data, 99.9)
    data = ((np.clip(data, imin, imax) - imin) / (imax - imin))
    return data


def normalize(img):
    ind = np.where(img != 0)
    mask = np.zeros(img.shape, dtype=np.int)
    mask[ind] = 1
    mean = img[ind].mean()
    std = img[ind].std()
    denominator = np.reciprocal(std)
    img = (img - mean) * denominator
    return mask * img


class Data(Dataset):
    def __init__(self, root, name, usenumber, istrain):
        self.root = root
        self.name = name
        self.usenumber = usenumber
        self.size = [512, 1024, 12]
        self.sizesmall = [192, 192, 12]
        self.pad = Compose([PadIfNeeded(shape=[self.size[0], self.size[1]], always_apply=True)], p=1)
        self.process = Compose([CropNonEmptyMaskIfExists(shape=self.size, always_apply=True, GetParams=True)], p=1.0)

        self.istrain = istrain
        if self.istrain:
            self.compose = Compose([
                GaussianNoise(var_limit=(0, 0.1), mean=0, p=0.5),
                RandomScale2(scale_limit=[0.7, 1.0], p=0.5),
                RotatePseudo2D(axes=[0, 1], p=0.5),
                ElasticTransformPseudo2D(alpha=1, sigma=10, alpha_affine=1, p=0.5)
            ],
                p=0.6)
        self.get3d = Compose([CropNonEmptyMaskIfExistsBalance(self.sizesmall,fg_class_number=19, always_apply=True, GetParams=True)], p=1.0)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, item):
        name = self.name[item]
        img = nib.load(os.path.join(os.path.join(self.root, 'image'), name))
        img_ = img.get_fdata()
        mask = nib.load(os.path.join(os.path.join(self.root, 'mask'), name))
        mask_ = mask.get_fdata()
        # print(np.max(mask))


        ## data augmentation
        if self.istrain:
            data = {'image': img_, 'mask': mask_}
            data = self.compose(**data)
            img_ = data['image']
            mask_ = data['mask']


        ## pad to size 512 * 1024
        data = {'image': img_, 'mask': mask_}
        data = self.pad(**data)
        img_ = data['image']
        mask_ = data['mask']


        ## extract size 512*1024*12
        data = {'image': img_, 'mask': mask_}
        data = self.process(**data)
        img = data['image']
        mask = data['mask']
        cutloc = data['params']

        ## downsample for 2d
        img_2d = resize(img_, output_shape=[img_.shape[0] // 2, img_.shape[1] // 2, img_.shape[-1]], order=1, clip=False, preserve_range=True, anti_aliasing=True)
        img2duse = img_2d[cutloc['x_min'] // 2: cutloc['x_max']// 2 , cutloc['y_min'] // 2: cutloc['y_max'] // 2, cutloc['z_min'] : cutloc['z_max']]


        ## pad up and down for 2d model
        minz = cutloc['z_min']
        maxz = cutloc['z_max']
        if minz == 0:
            downimg = np.expand_dims(
                np.concatenate([img_2d[cutloc['x_min'] // 2: cutloc['x_max']// 2 , cutloc['y_min'] // 2: cutloc['y_max'] // 2, 0:1],
                                img_2d[cutloc['x_min'] // 2: cutloc['x_max'] // 2, cutloc['y_min'] // 2: cutloc['y_max'] // 2,
                                0: self.size[-1] - 1]], axis=-1), axis=0)
        else:
            downimg = np.expand_dims(img_2d[cutloc['x_min'] // 2: cutloc['x_max'] // 2, cutloc['y_min'] // 2: cutloc['y_max'] // 2,
                                     minz - 1: minz - 1 + self.size[-1]], axis=0)
        if maxz == img_2d.shape[-1]:
            upimg = np.expand_dims(np.concatenate([img_2d[cutloc['x_min'] // 2: cutloc['x_max'] // 2,
                                                   cutloc['y_min'] // 2: cutloc['y_max'] // 2,
                                                   img_2d.shape[-1] - self.size[-1] + 1: img_2d.shape[-1]]
                                                      , img_2d[cutloc['x_min'] // 2: cutloc['x_max'] // 2,
                                                        cutloc['y_min'] // 2: cutloc['y_max'] // 2,
                                                        img_2d.shape[-1] - 1: img_2d.shape[-1]]], axis=-1), axis=0)
        else:
            upimg = np.expand_dims(img_2d[cutloc['x_min'] // 2: cutloc['x_max'] // 2, cutloc['y_min'] // 2: cutloc['y_max'] // 2,
                                   maxz + 1 - self.size[-1]: maxz + 1], axis=0)

        img_ = np.concatenate([downimg, np.expand_dims(img2duse, axis=0), upimg], axis=0)
        img_ = np.transpose(img_, axes=[3, 0, 1, 2])


        patch = []
        patchmask = []
        params = []
        for i in range(self.usenumber):
            data = {'image': img, 'mask': mask}
            data = self.get3d(**data)
            patch.append(np.expand_dims(data['image'], axis=0))
            patchmask.append(data['mask'])
            params.append(data['params'])

        patch = np.asarray(patch)
        patchmask = np.asarray(patchmask)

        # plt.subplot(121)
        # plt.imshow(mask[:,:,6])
        # plt.subplot(122)
        # plt.imshow(img[:,:,6])
        # plt.show()

        patch = torch.from_numpy(patch).type(torch.float32)
        patchmask = torch.from_numpy(patchmask).long()
        img_ = torch.from_numpy(img_).type(torch.float32)

        return patch, patchmask, img_, params

    def normalize(self, x):
        ma = np.max(np.max(x, axis=0), axis=0)
        mi = np.min(np.min(x, axis=0), axis=0)
        return (x - mi) * 255.0 / (ma - mi)



def progress(istrain, epoch, model2d, model3d, optimizer, loader, writer):
    index = Index()
    total_loss = AverageMeter()
    Mean_dice = AverageMeter()

    # labels = np.empty(len_y)
    # predicts = np.empty(len_y)
    # score = np.empty(len_y)
    for i, (x, y, whole, params) in enumerate(loader):
        if istrain:
            optimizer.zero_grad()
            model3d.zero_grad()
        # print(index(i,len(loader)-1), end='')
        x = x.float().squeeze(0).cuda()
        y = y.long().squeeze(0).cuda(1)
        whole = whole.float().squeeze(0).cuda(1)

        # print(torch.max(x))
        # print(torch.min(x))
        if istrain:
            model3d.train()
            model2d.eval()
        else:
            model3d.eval()
            model2d.eval()
        with torch.no_grad():
             locfeature, locscore = model2d(whole, params)
        locfeature = locfeature.cuda()
        locscore = locscore.cuda()
        output = model3d(x, locfeature,  locscore)
        Seg_one_hot = (one_hot(y.cpu(), 20)).permute(0, 4, 1, 2, 3)
        loss = criteion_ce_dc(output.cuda(1), torch.unsqueeze(y, 1))
        predict = torch.softmax(output.cpu().detach(), dim=1)
        Seg_prediction = torch.argmax(predict.cpu(), dim=1, keepdim=True).long()

        Seg_prediction_one_hot = torch.zeros((Seg_prediction.size(0), 20, Seg_prediction.size(2),
                                              Seg_prediction.size(3), Seg_prediction.size(4))).scatter_(1,
                                                                                                        Seg_prediction,
                                                                                                        1)

        # wt_dice = dice1(Seg_prediction_one_hot[:, 1, :, :]+Seg_prediction_one_hot[:, 2, :, :]+Seg_prediction_one_hot[:, 3, :, :],
        #                 (Seg_one_hot[:, 1, :, :]+ Seg_one_hot[:, 2, :, :]+ Seg_one_hot[:, 3, :, :]).cpu()).detach().numpy()
        #
        #
        # tc_dice = dice1(Seg_prediction_one_hot[:, 1, :, :]+Seg_prediction_one_hot[:, 3, :, :]
        #                 , (Seg_one_hot[:, 1, :, :]+Seg_one_hot[:, 3, :, :]).cpu()).detach().numpy()
        # et_dice = dice1(Seg_prediction_one_hot[:, 3, :, :], Seg_one_hot[:, 3, :, :].cpu()).detach().numpy()
        alldc = []
        for j in range(20):
            alldc.append(dc(Seg_one_hot[:, j, :, :].cpu().detach().numpy(),
                            Seg_prediction_one_hot[:, j, :, :].cpu().detach().numpy()))
        meandc = np.mean(np.asarray(alldc))

        # if y.size(0) == bs:
        #     start = i*bs
        #     endt = (i+1)*bs
        # else:
        #     start = i*bs
        #     endt = len_y
        # labels[start:endt] = y.cpu().numpy()
        # predicts[start:endt] = predict.cpu().numpy()
        # score[start:endt] = pred.detach().cpu().numpy()
        if istrain:
            loss.backward()
            nn.utils.clip_grad_norm_(model3d.parameters(), 12)
            # for i, (name, parms) in enumerate(model.named_parameters()):
            #     if i == 0:
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
            #               ' -->grad_value:', parms.grad)
            #         break
            optimizer.step()
        total_loss.update(loss.item() * y.size(0), y.size(0))
        Mean_dice.update(meandc * y.size(0), y.size(0))

        # if not istrain:
        #     if epoch % 5 == 0  and i == 1:
        #         pred = make_grid(Seg_prediction, test_batch_size)
        #         mask = make_grid(torch.unsqueeze(y.cpu(), dim=1), test_batch_size)
        #         writer.add_image('pred', pred, epoch)
        #         writer.add_image('mask', mask, epoch)

    writer.add_scalar('loss', total_loss.avg, epoch)
    writer.add_scalar('MeanDice', Mean_dice.avg, epoch)

    if istrain:
        print('epoch:{0} Trainloss:{1}, Meandice:{2}'.format(epoch, total_loss.avg, Mean_dice.avg))
    else:
        print('epoch:{0} validloss:{1}, Meandice:{2}'.format(epoch, total_loss.avg, Mean_dice.avg))
    return total_loss.avg, Mean_dice.avg


def readtxt(name):
    allname = []
    alllabel = []
    with open(name) as f:
        for row in f:
            allname.append(row.split(' ')[0])
            alllabel.append(int(row.split(' ')[1].split('\n')[0]))
    return allname, np.asarray(alllabel)


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


def writelog(path, filename, epoch, trainloss, validloss):
    with open(os.path.join(path, filename), 'a') as f:
        f.write(str(epoch) + ' ' + str(trainloss) + ' ' + str(validloss) + '\n')


def smoothVal(nowval, lastsmoothval=None):
    if lastsmoothval == None:
        return nowval
    else:
        return 0.9 * lastsmoothval + 0.1 * nowval


def caculate_gpu(x):
    gpunumber = len(x.split(','))
    return gpunumber

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

    def __init__(self,  num_classes, encoder='se_resnetx50', mode='cat'):
        super(DeeplabV3Plus3D, self).__init__()
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='number fold to train.')
    parser.add_argument('--gpuid', type=str, default="0,1", help='gpuid, at least three.')
    parser.add_argument('--datapath', type=str, default='./dataset/process3Ddata', help='data path.')
    parser.add_argument('--train_batch_size', type=int, default=4, help='batch size, which was crop in a sample, every iteration')
    parser.add_argument('--test_batch_size', type=int, default=6, help='batch size, which was crop in a sample, every iteration')
    parser.add_argument('--epochs', type=int, default=150, help='total number epoch')
    parser.add_argument('--branch_best', type=str, default="branch2", help='best_branch, which was the best model in 2D, default branch2.')
    parser.add_argument('--exid2D', type=str, default="ex0", help='experiment id for the 2D model for pretrain.')
    parser.add_argument('--exid', type=str, default="ex1", help='experiment id.')
    parser.add_argument('--seed', type=int, default=100, help=' random seed.')
    config = parser.parse_args()

    weightpath = os.path.join('./weight',  config.exid)
    logpath = os.path.join('./log',  config.exid)
    losspath = os.path.join('./loss',  config.exid)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    div = ['sub0', 'sub1', 'sub2', 'sub3', 'sub4']
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

            model2d = DualNetwork(None).cuda(len(config.gpuid.split(','))-1)
            model2d = load_GPUS(model2d, os.path.join(os.path.join(os.path.join('./weight', config.exid2D), sub), config.branch_best))

            model3d = DeeplabV3Plus3D(num_classes=20,
                                    encoder='se_resnetx50', mode='cat').cuda()
            model3d = nn.DataParallel(model3d, device_ids=[0])
            optimizer = torch.optim.Adam(model3d.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 100], last_epoch=-1)

            with open('./splitdataset.pkl', 'rb') as f:
                dataset = pk.load(f)
            trainfile = dataset[i][0]
            validfile = dataset[i][1]

            trainset = Data(root=config.datapath, name=trainfile, usenumber=config.train_batch_size, istrain=True)
            testset = Data(root=config.datapath, name=validfile, usenumber=config.test_batch_size, istrain=False)

            trainloader = DataLoader(trainset, batch_size=1, shuffle=True, pin_memory=True,
                                     num_workers=12)
            testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True, num_workers=12)

            criteion_ce_dc = dice_loss.DC_and_CE_loss(soft_dice_kwargs=dict(), ce_kwargs=dict())

            trainloss = []
            trainacc = []
            testloss = []
            testacc = []
            bestdsc = 0
            # smootheval = None
            for epoch in range(config.epochs):
                loss, _ = progress(istrain=True, epoch=epoch,model2d=model2d,  model3d=model3d, optimizer=optimizer,
                                   loader=trainloader, writer=train_writer)
                scheduler.step()
                trainloss.append(loss)
                with torch.no_grad():
                    vloss, dsc = progress(istrain=False, epoch=epoch, model2d=model2d, model3d=model3d, optimizer=optimizer,
                                          loader=testloader, writer=test_writer)
                testloss.append(vloss)
                bestdsc = savemodel(os.path.join(weightpath, sub), model3d, dsc, bestdsc)
