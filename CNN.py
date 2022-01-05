import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.model_selection import train_test_split


data=pd.read_csv("LogisticRegData/train.csv",dtype=(np.float32))

target_numpy=data.label.values
features_numpy=data.iloc[:,data.columns!="label"].values/255 #"Normalization"

features_train,features_test,target_train,target_test=train_test_split(features_numpy,target_numpy,test_size=0.2,random_state=42)

# Float from double

features_train=features_train.astype(np.float32)
target_train=target_train.astype(np.float32)
features_test=features_test.astype(np.float32)
target_test=target_test.astype(np.float32)

# Tensor from numpy

featurestrain=torch.from_numpy(features_train)
targettrain=torch.from_numpy(target_train).type(torch.LongTensor)
featurestest=torch.from_numpy(features_test)
targettest=torch.from_numpy(target_test).type(torch.LongTensor)

# Batch_Size ,Epoch

batch_size=100
n_iter=2500
n_epoch=n_iter/(len(features_train)/batch_size)
n_epoch=int(n_epoch)

# Verimi Pytorcha uyumlu hale getirme işlemlerim.
train=torch.utils.data.TensorDataset(featurestrain,targettrain)
test=torch.utils.data.TensorDataset(featurestest,targettest)
# DataLoader

train_loader=DataLoader(train,batch_size,shuffle=False)
test_loader=DataLoader(test,batch_size,shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        
    # Convolution 1 
    #RuntimeError: mat1 and mat2 shapes cannot be multiplied (800x100 and 512x10) Böyle bir hatada katmanların çıktısında sorun var .
    
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        
        # flatten
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        
        self.feature_extractor=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=(1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6,16,5,stride=(1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=1),
            nn.Conv2d(16, 120, kernel_size=5,stride=1),
            nn.ReLU()
            )
        self.classifier=nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
            
            )
    def forward(self,x):
        out=self.feature_extractor(x)
        out=torch.flatten(out,1)
        logits = self.classifier(out)
        probs = F.softmax(logits, dim=1)
        return logits, probs


model=Model()

error=nn.CrossEntropyLoss()

lr=0.002

optimer=torch.optim.Adam(model.parameters(),lr=lr)


loss_list=[]
iter_list=[]
acc_list=[]
count=0

for epoch in range(n_epoch):
    for i,(images,labels) in enumerate(train_loader):
        train_data=Variable(images.view(100,1,28,28)) # Eğer Burada shape '[100, 1, 28, 28]' is invalid for input of size 200704 Hatası alıyorsan batchsize ile oyna
        
        labels=Variable(labels)
        
        optimer.zero_grad()
        
        outputs=model(train_data)
        
        loss=error(outputs,labels)
        
        loss.backward()

        optimer.step()
        
        count+=1
        
        if count%50==0:
            correct=0
            total=0
            for i,(images,labels) in enumerate(test_loader):
                test=Variable(images.view(100,1,28,28))
                outputs=model(test)
                predicted=torch.max(outputs,1)[1]
                
                total+=len(labels)
                
                correct+=(predicted==labels).sum()
            acc=100*correct/float(total)
            iter_list.append(count)
            loss_list.append(loss.data)
            acc_list.append(acc)
            if count%500==0:
                
                print("acc {} loss {} iter {}".format(acc,loss.data,count))
                
plt.plot(iter_list,loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
# visualization accuracy 
plt.plot(iter_list,acc_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()