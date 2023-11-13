import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
from torchvision.transforms import *
from torch.utils.data import DataLoader
import numpy as np
import torchvision.models as mds
import pytorch_warmup as warmup
import sys
import config
from qqdm import qqdm
from torchvision import models

transform=transforms.Compose(
    [
    transforms.RandomResizedCrop((224,224)),
    transforms.RandomVerticalFlip(p = 0.5),
    transforms.RandomAffine((0,180),(0.1,0.2)),
    transforms.ToTensor(),
    ]
)


data_dir = config.image_path #data path
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_train.head()

# load Dataset
image_folder = os.path.join(data_dir,"train_images")


class LEAFDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id

        image_file = os.path.join(image_folder, img_id)
        image = Image.open(image_file)

        if self.transforms is not None:
            image = self.transforms(image)
        image = np.array(image).astype(np.float32)
        image /= 255

        return torch.tensor(image), torch.tensor(row.label)

model = None
if config.model == 'resnet':
    model = models.resnet50(True)
    model.fc = nn.Linear(2048,5)
    print(model)
elif config.model == 'resnext':
    model =models.resnext50_32x4d(True)
    model.fc = nn.Linear(2048,5)
    print(model)

#train model

true_label=[]
pre_label=[]
learning_rate = config.lr
EPOCH = config.epoch

# change optimizer
opti = None
if config.model == "resnext":
    opti = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
elif config.model == "resnet":
    opti = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
loss_function=nn.CrossEntropyLoss()
choose = 0
def train(*,epoch, model, train_dataset, loss_function, optimizer):
    sum_loss = 0
    for i,data in enumerate(train_dataset):
        optimizer.zero_grad()
        features, labels = data
        features=features.cuda()
        labels=labels.cuda()
        out = model(features)
        loss = loss_function(out, labels)
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
    return float(sum_loss/len(train_dataset))


def test(*, model, test_dataset, train_dataset):
    def check(dataset):
        right = 0
        total = 0
        with torch.no_grad():
            for data in dataset:
                image, label = data
                true_label.append(label)
                image = image.cuda()
                label = label.cuda()
                out = model(image)
                total = total+out.size(0)
                _, index = torch.max(out.data, dim=1)
                pre_label.append(index)
                right += (index == label).sum().item()
        return float(right / total)
    return check(train_dataset), check(test_dataset)
print("done")

full_dataset=LEAFDataset(df_train,transform)
size = int(0.1*len(full_dataset))
split_size = [size for i in range(9)]
split_size.append(len(full_dataset)-sum(split_size))
Datasets = torch.utils.data.random_split(
    full_dataset,split_size,generator=torch.Generator().manual_seed(42))
for idx,d in enumerate(Datasets):
  print("Dataset {} : length : {}".format(idx,len(d)))

BATCH_SIZE  = config.batch_size
DataLoaders = [DataLoader(d,batch_size=BATCH_SIZE,shuffle=True) for d in Datasets]

def ChooseDataset(index):
  test_dataset = None
  train_datasets = []
  for i in range(len(DataLoaders)):
    if index == i:
      test_dataset = DataLoaders[i]
    else:
      train_datasets.append(DataLoaders[i])
  return train_datasets,test_dataset

#Setting parameters

train_acc_list = []
test_acc_list  =[]
loss_list =[]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
Bar  = qqdm(range(EPOCH))
for i in Bar:
    if choose % 10 == 0:
        choose = 0
    train_datasets,test_dataset = ChooseDataset(choose)
    choose+=1
    train_loss = 0.0
    for train_dataset in train_datasets:
      train_loss += train(epoch = i+1,model = model,train_dataset=train_dataset,loss_function=loss_function,optimizer=opti)
    train_acc,test_acc = test(model=model,test_dataset=test_dataset,train_dataset=train_datasets[np.random.randint(0,8)])
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    loss_list.append(float(train_loss/9))
    Bar.set_infos({"epoch":i+1,"train acc":train_acc,"test acc":test_acc,"loss":float(train_loss/9)})

# save train model
# edit your save model address
# path = r"C:\Users\HARRY\Desktop\Resnext50\Resnext_model.pth"
torch.save(model.state_dict(),config.save_path)