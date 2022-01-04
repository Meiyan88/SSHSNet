import os
import torch
import matplotlib
# matplotlib.use('AGG')#
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,brier_score_loss
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch

import random


class CaptchaCreator(torch.utils.data.Sampler):
    def __init__(self, count=200, usecount=250, bs=2):
        self.count = count
        self.usecount = usecount
        self.bs = bs
    # @staticmethod
    def random_seq(self):
        # 将其中的choice_seq，count 改为你需要的参数
        return [random.choice(list(range(self.count))) for _ in range(self.usecount * self.bs)]

    def shuffle(self):
        digits = self.random_seq()
        random.shuffle(digits)
        return digits

    def __iter__(self):
        return iter(self.shuffle())
    def __len__(self):
        return len(self.shuffle())

class EarlyStopping:
    '''Early stop the training if validation loss doesn't improve after
    a given patience'''

    def __init__(self, patience=7, verbose=False,
                 delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''save model when validation loss decreas'''
        if self.verbose:
            print(f'validation loss decrease ({self.val_loss_min:.6f})')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def loadpth(path, model, old, new):
    file = os.listdir(path)
    if len(file) != 0:
        for name in file:
            if endWith(name, '.pkl'):
                model.load_state_dict(
                    torch.load(os.path.join(path, name), map_location={'cuda:' + str(old): 'cuda:' + str(new)}))
    return model


def endWith(s, *endstring):
    array = map(s.endswith, endstring)
    if True in array:
        return True
    else:
        return False


def plot_auc(savepath, name, y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    # linewidth
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(savepath , name + '.png'))
    plt.close()

def plot_calib_curve(savepath,name,y_test,y_pred):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred, n_bins=10)
    clf_score = brier_score_loss(y_test, y_pred, pos_label=y_test.max())
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    # plt.plot(mean_predicted_value,fraction_of_positives,"s-")
    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s (%1.3f)" % (name, clf_score))
    plt.ylabel("Fraction of positives")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title('Calibration plots  (reliability curve)')
    plt.savefig(os.path.join(savepath, name + '.png'))
    plt.close()




def testresult(model, loader, len_y, bs):
    labels = np.empty(len_y)
    score = np.empty(len_y)
    predicts = np.empty(len_y)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if len(x.size()) == 5:
                x = x.cuda().float()
                # print(x.device)
                y = torch.squeeze(y, dim=-1).cuda().long()
                model.eval()
                output = model(x)
                # loss = criterion(output,y)
                _, predict = torch.max(torch.log_softmax(output, 1), 1)
                pred = torch.softmax(output, 1)[:, 1]
            else:
                x = torch.squeeze(x, dim=0).cuda()
                y = torch.squeeze(y, dim=0).cuda().long()
                model.eval()
                output = model(x)
                predict = torch.argmax(torch.mean(torch.softmax(output, 1), dim=0), dim=0)
                pred = torch.mean(torch.softmax(output, 1), dim=0)[1]
                y = torch.unsqueeze(torch.mean(y.float(), dim=0).long(), dim=0)


            if y.size(0) == bs:
                start = i * bs
                endt = (i + 1) * bs
            else:
                start = i * bs
                endt = len_y
            labels[start:endt] = y.cpu().numpy()
            score[start:endt] = pred.detach().cpu().numpy()
            predicts[start:endt] = predict.cpu().numpy()
    return labels, score, predicts

def extract_feature(path,model, loader, len_y, bs):
    feature = np.empty((len_y,1380))
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.cuda().float()
            model.eval()
            hidden = model(x)
            if y.size(0) == bs:
                start = i * bs
                endt = (i + 1) * bs
            else:
                start = i * bs
                endt = len_y
            feature[start:endt] = hidden.cpu().numpy()
    np.save(os.path.join(path,'featureavg.npy'),feature)






class Index(object):
    def __init__(self, number=50, decimal=2):
        """
        :param decimal: 你保留的保留小数位
        :param number: # 号的 个数
        """
        self.decimal = decimal
        self.number = number
        self.a = 100 / number  # 在百分比 为几时增加一个 # 号

    def __call__(self, now, total):
        # 1. 获取当前的百分比数
        percentage = self.percentage_number(now, total)

        # 2. 根据 现在百分比计算
        well_num = int(percentage / self.a)
        # print("well_num: ", well_num, percentage)

        # 3. 打印字符进度条
        progress_bar_num = self.progress_bar(well_num)

        # 4. 完成的进度条
        result = "\r%s %s" % (progress_bar_num, percentage)
        return result

    def percentage_number(self, now, total):
        """
        计算百分比
        :param now:  现在的数
        :param total:  总数
        :return: 百分
        """
        return round(now / total * 100, self.decimal)

    def progress_bar(self, num):
        """
        显示进度条位置
        :param num:  拼接的  “#” 号的
        :return: 返回的结果当前的进度条
        """
        # 1. "#" 号个数
        well_num = "#" * num

        # 2. 空格的个数
        space_num = " " * (self.number - num)

        return '[%s%s]' % (well_num, space_num)