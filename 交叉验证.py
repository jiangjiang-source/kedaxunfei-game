import torch
import torch.optim as optim
from PIL import Image
torch.backends.cudnn.deterministic=False
torch.backends.cudnn.benchmark=True
import torchvision.transforms as transform
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
import cv2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
device="cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)
torch.manual_seed(6)
torch.cuda.manual_seed_all(42)

a=np.random.choice(np.arange(100))

class mydataset(Dataset):
    def __init__(self,x1_data,x2_data):
        self.x1_data = x1_data
        self.x2_data=x2_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self,idx):
        img1 = self.x_data[idx]
        y1 = self.y_data[idx]
        if np.random.rand() < 0.5:
            idx2 = np.random.choice(np.arange(len(self.y_data))[self.y_data==y1],1)
        else:
            idx2 = np.random.choice(np.arange(len(self.y_data))[self.y_data!=y1],1)
        img2 = self.x_data[idx2[0]]
        y2 = self.y_data[idx2[0]]
        label = 0 if y1==y2 else 1
        return img1,img2,label




class QRdataset(Dataset):
    def __init__(self,train_path,label,transform=None):
        self.train_path=train_path
        self.label=label
        if transform is not None:
            self.transform=transform
        else:
            self.transform=None
    def __getitem__(self,index):
        path=self.train_path[index]
        label = self.label[index]
        img1=Image.open(path).convert('RGB')
        img1 = np.array(img1)
        img1 = cv2.resize(img1, (512, 512))
        img = np.array(img1)
        img=self.transform(img)
        return img,label

    def __len__(self):
        return len(self.train_path)


class QRdataset1(Dataset):
    def __init__(self,train_path,transform=None):
        self.train_path=train_path
        if transform is not None:
            self.transform=transform
        else:
            self.transform=None
    def __getitem__(self,index):
        path=self.train_path[index]
        img1=Image.open(path).convert('RGB')
        img1 = np.array(img1)
    #    img1 = np.where(img1 > 20, img1, 0)
        # A, B, C = img1.shape
        # IMG1 = img1[:int(A / 2), :int(B / 2), :]
        # IMG2 = img1[:int(A / 2), int(B / 2):, :]
        # IMG3 = img1[int(A / 2):, :int(B / 2), :]
        # IMG4 = img1[int(A / 2):, int(B / 2):, :]
        # img1 = (IMG1 + IMG2 + IMG3 + IMG4)

       # img1 = cv2.convertScaleAbs(cv2.Sobel(img1, cv2.CV_64F, 1, 1))

        img1= cv2.resize(img1, (512, 512))
        img=np.array(img1)
        img=self.transform(img)
        return img
    def __len__(self):
        return len(self.train_path)


testdata=QRdataset1(train_path=test_feature,transform=transform.Compose([
    transform.ToTensor(),
    # transform.RandomCrop((224, 224)),
  #  transform.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.5),
    transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

import torch
from torch import nn
import torch.nn.functional as F

class Arc(nn.Module):
    def __init__(self,feature_dim=2,cls_dim=10):
        super(Arc, self).__init__()
        #x是（N，V）结构，那么W是（V,C结构），V是特征的维度，C是代表类别数
        self.W = nn.Parameter(torch.randn(feature_dim,cls_dim))

    def forward(self,feature,m=1,s=10):
        x = F.normalize(feature,dim=1)
        w = F.normalize(self.W,dim=0)
        cos = torch.matmul(x,w)/10 #（N,C）
        a = torch.acos(cos) #(N,C)
        top = torch.exp(s*torch.cos(a+m))  #(N,C)
        down = torch.sum(torch.exp(s*torch.cos(a)),dim=1,keepdim=True)-torch.exp(s*torch.cos(a))  #第一项(N,1)  keepdim=True保持形状不变.这是我们原有的softmax的分布。第二项(N,C),最后结果是(N,C)
        out = torch.log(top/(top+down))  #(N,C)
        return out
class ThreeNet(torch.nn.Module):
    def __init__(self):
        super(ThreeNet,self).__init__()
        resnet=models.resnet18(pretrained=True)
        numFit = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(numFit, 32)
        self.model=resnet
        self.out=Arc(feature_dim=32,cls_dim=2)
    def forward(self,img):
        out=self.model(img)
        out=self.out(out)
        return out
model=ThreeNet().to(device)

for name, para in model.named_parameters():
    # 除最后的全连接层外，其他权重全部冻结
    if 'fc' in name or 'out' in name:
        para.requires_grad_(True)
    else:
        para.requires_grad_(False)
   # print(name,para.requires_grad)
        # 或者 para.requires_grad = False



torch.save(model,'first.pth')
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameters: %.2fM" % (total/1e6))

#cretiation=torch.nn.CrossEntropyLoss()
cretiation=torch.nn.NLLLoss()
class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].cuda() # 注意，这里的alpha是给定的一个list(tensor
#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
# 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

focalloss=MultiCEFocalLoss(class_num=2)
skf = KFold(n_splits=5,shuffle=True,random_state=42)



for idx, (train_index, val_index) in enumerate(skf.split(train_feature, train_label)):
    d = 0
    # if idx<=3:
    #     continue
    model = torch.load('first.pth')
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)#, momentum=0.95)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.00001 , momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    adam=optim.Adam(model.parameters(),lr=0.005,)

    train_feature=np.array(train_feature)
    train_label=np.array(train_label)
    traindataset=QRdataset(train_path=train_feature[train_index],label=train_label[train_index],transform=transform.Compose([
                           transform.ToTensor(),
                            # transform.RandomHorizontalFlip(p=0.01),
                            # transform.RandomVerticalFlip(p=0.01),
                            # transform.RandomCrop((224, 224)),
                            transform.RandomRotation(degrees=(-45,45)),
                          #  transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    valdataset=QRdataset(train_path=train_feature[val_index],label=train_label[val_index],transform=transform.Compose([
                          transform.ToTensor(),
                           # transform.RandomCrop((224, 224)),
                    #     transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                          transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    val_loader = DataLoader(valdataset, batch_size=12, shuffle=False, drop_last=False, num_workers=24)
    test_loader = DataLoader(testdata, batch_size=12, shuffle=False, drop_last=False, num_workers=24)
    train_loader = DataLoader(traindataset, batch_size=8, shuffle=True, drop_last=False, num_workers=24)

    for epoch in range(50):
        model.train()
        LOSS=0
        if epoch<10:
            cretiation = torch.nn.CrossEntropyLoss()
        # else:
        #     cretiation=MultiCEFocalLoss(class_num=2)
        for i ,(img,label) in enumerate(train_loader):
            img = img.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)
            l = np.random.beta(1, 1)
            index = torch.randperm(img.size(0))
            image_a, image_b = img, img[index]
            label_a, label_b = label, label[index]
            mix_image = l * image_a + (1 - l) * image_b
            output = model(mix_image)
            loss = l * cretiation(output, label_a.long()) + (1 - l) * cretiation(output, label_b.long())
            LOSS+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if i % 100== 0:
            #      print(LOSS/(i+1), '学习率:', optimizer.param_groups[0]['lr'])
        print(LOSS, '学习率:', optimizer.param_groups[0]['lr'])
        scheduler.step()

        if epoch %1== 0:
            model.eval()
            for i, (img, label) in enumerate(train_loader):
                img = img.to(torch.float32).to(device)
                label = label.to(torch.float32).numpy()
               # output = torch.softmax(model(img), dim=-1)
                output = model(img)
                output = output.cpu().detach().numpy()
                pre= np.argmax(output,axis=1) if i==0 else np.hstack((pre,np.argmax(output,axis=1) ))
                target = label if i == 0 else np.hstack((target, label))
            print(epoch,'训练集acc:', accuracy_score(target, pre),'f1=',f1_score(target, pre))

        if epoch % 1 == 0:
            model.eval()
            for i, (img, label) in enumerate(val_loader):
                img = img.to(torch.float32).to(device)
                label = label.to(torch.float32).numpy()
                output = model(img)
         #       output = torch.softmax(output, dim=-1)
                output = output.cpu().detach().numpy()
                pre = np.argmax(output, axis=1) if i == 0 else np.hstack((pre, np.argmax(output, axis=1)))
                target = label if i == 0 else np.hstack((target, label))
       #     print(111111111111111111111111111111111111111111111111111111111111111111, )
            print(np.sum(pre),epoch, '验证集acc:', accuracy_score(target, pre), 'f1=', f1_score(target, pre))

        if d <= f1_score(target, pre):
            d = f1_score(target, pre)
            model.eval()
            for i, img in enumerate(test_loader):
                img = img.to(torch.float32).to(device)
                output = model(img)
              #  output = torch.softmax(output, dim=-1)
                output = output.cpu().detach().numpy()
                pre = np.argmax(output, axis=1) if i == 0 else np.hstack((pre, np.argmax(output, axis=1)))
                pre1 = np.max(output, axis=1) if i == 0 else np.hstack((pre1, np.max(output, axis=1)))
                # target = label if i == 0 else np.hstack((target, label))
            print(11111111111111, np.sum(pre))
            data['label'] = pre
            data['acc'] = pre1
            # data=pd.merge(data,sub,on='image')
            modeladdress = '/home/jiang/桌面/game1/model' + str(idx) + '.pth'
            sub1 = '/home/jiang/桌面/game1/sub' + str(idx) + '.csv'
            torch.save(model, modeladdress)
            data.to_csv(sub1, index=None)