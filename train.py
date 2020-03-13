
import argparse
import cv2
import json
import random
from ranger import Ranger
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import sys
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import traceback
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.transform import rotate, resize
from numpy import fliplr, flipud



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()  

        ''' declare layers used in this network'''
        self.net = models.resnet18(pretrained=True)
        self.net = torch.nn.Sequential(*(list(self.net.children())[:-1]))#.to('cuda:0')
#        for p in self.net.parameters():
#            p.requires_grad = False
#        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
#        self.fc3 = nn.Linear(1024, 1024)
#        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 3)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
#        self.prelu3 = nn.PReLU()
#        self.prelu4 = nn.PReLU()
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
#        BatchSize = x.shape[0]
#        x = self.bn1(x)
        x = self.net(x).squeeze()#.view(BatchSize,-1)
        x = self.bn2(x)
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
#        x = self.prelu3(self.fc3(x))
#        x = self.prelu4(self.fc4(x))
        x = self.softmax(self.fc5(x))
        return x
def arg_parse():
    parser = argparse.ArgumentParser(description='COPD')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='data', 
                    help="root path to data directory")
    parser.add_argument('--workers', default=0, type=int,
                    help="number of data loading workers (default: 4)")
    
    # training parameters
    parser.add_argument('--gpu', default=0, type=int, 
                    help='In homework, please always set to 0')
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=16, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.0001, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.001, type=float,
                    help="initial learning rate")
    
    # resume trained model
    parser.add_argument('--resume', type=str, default='', 
                    help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='model')
    parser.add_argument('--random_seed', type=int, default=999)

    args = parser.parse_args()

    return args


def main(args):
    
    
#    img = plt.imread(train_img_list[0])
#    plt.imshow(img)
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ''' setup GPU '''
    #    torch.cuda.set_device(args.gpu)
    
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iters = 0
    
    
    best_acc = 0
    best_recall = 0
    train_loss=[]
    total_wrecall = [0]
    total_acc = [0]
    
    ''' load dataset and prepare data loader '''
    
    print('===> loading data')
    train_set = MyDataset(r'E:\ACV\MangoClassify', 'C1-P1_Train', 'train.csv','train')
    test_set = MyDataset(r'E:\ACV\MangoClassify', 'C1-P1_Dev', 'dev.csv','test')
#    print(train_set[0])
    train_loss = []
    print('===> build dataloader ...')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=args.train_batch,num_workers=args.workers,shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=args.test_batch,num_workers=args.workers,shuffle=False)

    
    
    ''' load model '''
    print('===> prepare model ...')
    model = CNN().to(device)
    
    
    ''' define loss '''
    criterion = nn.CrossEntropyLoss()
    
    ''' setup optimizer '''
#    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = Ranger(model.parameters(), lr=args.lr)    
    ''' train model '''
    print('===> start training ...')
    
    for epoch in range(1, args.epoch+1):
#        model.train()
        for idx, (imgs, label) in enumerate(train_loader):
#            print(imgs, label)
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1
        
            ''' move data to gpu '''
        #            print('===> load data to gpu')
            imgs = imgs.permute(0, 3 ,1, 2).to(device, dtype = torch.float)
            label = label.to(device)
            
            ''' forward path '''
            pred = model(imgs)

        
            ''' compute loss, backpropagation, update parameters '''
#            print('===> calculate loss')
            loss = criterion(pred, label) # compute loss
            train_loss += [loss.item()]
            torch.cuda.empty_cache()
            optimizer.zero_grad()         # set grad of all parameters to zero
            loss.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters
            
            train_info += ' loss: {:.8f}'.format(loss.item())
        
            print(train_info)#, end="\r")
        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            test_info = 'Epoch: [{}] '.format(epoch)
            model.eval()
            
            correct = 0
            total = 0
            tp_A = 0
            tp_B = 0
            tp_C = 0
            fn_A = 0
            fn_B = 0
            fn_C = 0
#            loss = 0
            for idx, (imgs, label) in enumerate(test_loader):
                imgs = imgs.permute(0, 3 ,1, 2).to(device, dtype = torch.float)
                
#                gt = gt.to(device)
                pred = model(imgs)
#                torch.cuda.empty_cache()
#                loss += criterion(output, gt).item()
                a, b, c, d, e, f, g, h, i = confusion_matrix(label.detach().numpy(),pred.argmax(-1).cpu().detach().numpy()).ravel()
                tp_A += a
                fn_A += (a + b + c)
                tp_B += e
                fn_B += (e + d + f)
                tp_C += i
                fn_C += (i + g + h)
                correct += (a+e+i)
                total += len(label)
            acc = correct / total
            w_recall = ((tp_A/fn_A) + (tp_B/fn_B) + (tp_B/fn_B)) / 3
            total_wrecall += [w_recall]
            total_acc += [acc]
            test_info += 'Acc:{:.8f} '.format(acc)
            test_info += 'Recall:{:.8f} '.format(w_recall)
            
            print(test_info)
#            print(tn, fp, fn, tp)
        
            ''' save best model '''
            if w_recall > best_recall:
                best_recall = w_recall
                save_model(model, os.path.join(args.save_dir, 'model_best_recall.h5'))
                
            if acc > best_acc:
                best_acc = acc
                save_model(model, os.path.join(args.save_dir, 'model_best_acc.h5'))
                
                
    
        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}_acc={:8f}_recall={:8f}.h5'.format(epoch,acc,w_recall)))
        
    plt.figure()
    plt.plot(range(1,len(train_loss)+1),train_loss,'-')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.show()
    
    plt.figure()
    plt.plot(range(1,len(total_acc)+1),total_acc,'-')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Best Accuracy:" + str(best_acc))
    plt.show()
    
    plt.figure()
    plt.plot(range(1,len(total_wrecall)+1),total_wrecall,'-')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Best Recall:" + str(best_recall))
    plt.show()


    
    
def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)    
    
class MyDataset(Dataset):
    def __init__(self, dir_path , img_file, csv, state):
        self.dir_path = dir_path
        self.csv = csv
        self.img_file = os.path.join(dir_path, img_file)
        with open(os.path.join(self.dir_path,self.csv),'rb') as f:
            self.img_label_file = pd.read_csv(f, header=None)#.head(80)
        self.label_dict = {'A': 0,
                      'B': 1,
                      'C': 2}
        self.state = state
    def transform(self, image):
        
#        plt.figure()
#        plt.imshow(image)
#        plt.show()
        p = 0.5
        # Random horizontal flipping
        if random.random() > p:
            image = cv2.flip(image, 1)
        # Random vertical flipping
        if random.random() > p:
            image = cv2.flip(image, 0)

        if random.random() > p:
            image = cv2.GaussianBlur(image,(3,3),0)

        if random.random() > p:
            image = random_noise(image)

        return image
    def __len__(self):
       
        return self.img_label_file.shape[0]
    def __getitem__(self,idx):
        
        img_name = self.img_label_file.iloc[idx][0]
        
        
        image = cv2.imread(os.path.join(self.img_file, img_name))
        image = cv2.resize(image,(512,512))
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        image = image / np.max(image) * 255.
        if self.state == 'train':
            image = self.transform(image)
        label = self.label_dict[self.img_label_file.iloc[idx][1]]
        return torch.tensor(image.copy()).float(), torch.tensor(label)
    
if __name__ == '__main__':
    args = arg_parse()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)