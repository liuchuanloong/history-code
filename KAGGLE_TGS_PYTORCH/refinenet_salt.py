import numpy as np
from pytorch_refinenet.pytorch_refinenet import RefineNet4Cascade
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
from copy import deepcopy
from validation import val
import torch.nn.functional as F
import sys


MODEL_PATH = './model/'
LR = 0.01
EPOCH = 100
TRAIN_CSV = './data/train_split.csv'
VAL_CSV = './data/val_split.csv'
VALIDATE = True
N_CLASS = 2
CUDA = True

class SaltData(data.Dataset):

    def __init__(self, csv_path):

        self.n_class = 2
        self.csv_path = csv_path
        self.path_train = "./saltdata/train"
        self.train_ids = self.get_ids(self.csv_path)
        self.lengths = len(self.train_ids)


    def get_ids(self, csv_path):

        # list_id中保存的是训练图片的名称，但是没有后缀
        df = pd.read_csv(csv_path)
        self.list_id = []
        for i, item in df.iterrows():
            self.list_id.append(item[0])
        return self.list_id

    def transforms(self, im):

        im_tfs = transforms.Compose([transforms.Resize(96), transforms.ToTensor()])
        im = im_tfs(im)
        return im

    #     def dataset(self):

    #         print('Getting and resizing train images and masks ... ')
    #         sys.stdout.flush()
    #         self.imgs = []
    #         self.labels = []
    #         for n, id_ in tqdm_notebook(enumerate(self.train_ids), total=len(self.train_ids)):
    #             img = Image.open(self.path_train + '/images/' + id_)
    #             mask = Image.open(self.path_train + '/masks/' + id_)
    #             img, label = self.transforms(img, mask)
    #             self.imgs.append(img)
    #             self.labels.append(label)
    #         print('Done!')

    def __getitem__(self, index):

        id_ = self.train_ids[index]

        img = Image.open(self.path_train + '/images/' + id_).convert('RGB')
        mask = Image.open(self.path_train + '/masks/' + id_).convert('RGB')
        mask = np.resize(mask,(24,24))
        # mask = np.array(mask, dtype=np.int64)
        # label = torch.from_numpy(mask)

        mask = torch.from_numpy(mask.copy()).long()

        # create one-hot encoding
        h, w = mask.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][mask == c] = 1


        if self.transforms is not None:
            img = self.transforms(img)
        return img, target, mask

    def __len__(self):

        return len(self.train_ids)

def  train(VALIDATE=VALIDATE, CUDA=CUDA):
    """

    :param validate: bool, if True, validation dataset
    :param cuda: bool, if True, use cuda
    """
    train_dataset = SaltData(TRAIN_CSV)
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    if VALIDATE:
        val_dataset = SaltData(VAL_CSV)
        val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

    print('start load model!!!')
    if CUDA:
        net = RefineNet4Cascade((3, 96), num_classes=2).cuda()
        print(net)
        print('transfered to cuda done!!!')
    else:
        net = RefineNet4Cascade((3, 96), num_classes=2)
        print(net)
        print("Warning: no cuda model!!!")

    optimizer = torch.optim.SGD(net.parameters(), lr= 0.01, momentum=0.9, weight_decay=0.005)
    # criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter()

    for epoch in range(EPOCH):
        sys.stdout.flush()
        ts = time.time()
        for step, (b_x,b_y,_) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
            optimizer.zero_grad()
            if CUDA:
                inputs = torch.autograd.Variable(b_x).cuda()
                targets = torch.autograd.Variable(b_y).cuda()
            else:
                inputs = torch.autograd.Variable(b_x)
                targets = torch.autograd.Variable(b_y)
            outputs = F.relu(net(inputs))
            # print(outputs)
            # loss = criterion(outputs, targets)
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            loss.backward()
            optimizer.step()
            ts10 = time.time()
            if step % 10 == 0 and step != 0:
                in_cur_state = {
                    'state': 'continue',
                    'epoch': epoch + 1,
                    'iter': step + 1,
                    'state_dict':net.state_dict(),
                }
                writer.add_scalar('data/loss', loss.data[0], epoch)
                print("epoch{}, iter{}, loss: {}, time elapsed {}".format(epoch, step, loss.data[0], time.time() - ts10))
                model_path = MODEL_PATH + "epoch{}_iter{}.pth.tar".format(epoch, step)
                torch.save(deepcopy(in_cur_state), model_path)
        if VALIDATE:
            val(model=net, val_loader=val_loader, epoch = epoch, CUDA=CUDA)

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    cur_state = {
        'state':'finish',
        'epoch': epoch + 1,
        'state_dict':net.state_dict(),
    }
    model_path = MODEL_PATH + "refinenet_model.pth.tar"
    torch.save(deepcopy(cur_state), model_path)


if __name__ == '__main__':
    train()


