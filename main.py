import os
import argparse
import numpy as np
import pickle


import time
import copy

from sklearn.feature_extraction import FeatureHasher

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type = str, help = 'dataset name')
parser.add_argument('--hidden_dim', type = int, default = 10, help = 'num of hidden nodes')
parser.add_argument('--num_epoch', type = int, default = 20, help = 'num of epochs')
parser.add_argument('--learning_rate', type = float, default = 1e-2, help = 'learning rate')
parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
parser.add_argument('--val_split', type = float, default = 0.2, help = 'val_split')

args = parser.parse_args()

def make_Vocab(docs):
    word2index={'<unk>' : 0}
    for doc in docs:
        for token in doc:
            if word2index.get(token)==None:
                word2index[token]=len(word2index)
    return word2index

def make_Dict(tokens):
    Doc = dict()
    for t in tokens:
        try :
            Doc[t] += 1
        except:
            Doc[t] = 1
    return Doc

def split_Data(dataset,val_split,batch_size):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2, sampler = train_sampler)
    valid_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2, sampler = valid_sampler)

    dataloaders= {'train':train_loader,'val':valid_loader}

    return dataloaders

class TrainDataset(Dataset):
    
    def __init__(self):

        self.H = Hasher.transform(train_Xd)
        self.y_data = torch.from_numpy(np.array(dataDict['train_Y']))
        
    def __getitem__(self,index):
        return torch.from_numpy(self.H[index].toarray()[0]), self.y_data[index]
    
    def __len__(self):
        return self.H.shape[0]


class TestDataset(Dataset):
    
    def __init__(self):

        self.H = Hasher.transform(test_Xd)
        self.y_data = torch.from_numpy(np.array(dataDict['test_Y']))
        
    def __getitem__(self,index):
        return torch.from_numpy(self.H[index].toarray()[0]), self.y_data[index]
    
    def __len__(self):
        return self.H.shape[0]  

class TextClassifier(nn.Module):
    def __init__(self,vocab_size,hidden_dim,num_class):
        super(TextClassifier,self).__init__()
        #self.embedding = nn.EmbeddingBag(vocab_size, hidden_dim, sparse=True)
        self.input_layer = nn.Linear(vocab_size,hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, num_class)
        self.output_layer = nn.Softmax(dim=1)
        
    def forward(self,text):
        embedded = self.input_layer(text)
        output = self.hidden_layer(embedded)
        return self.output_layer(output)


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train','val']:
            if phase == 'train':
                
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            count = 0

            for inputs,labels in dataloaders[phase]:
                count += len(inputs)
                inputs,labels = inputs.to(device),labels.to(device)
                inputs,labels = Variable(inputs.float()), Variable(labels)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.float())
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase =='train':
                        loss.backward()
                        optimizer.step()
                        
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                            
            epoch_loss = running_loss / count
            epoch_acc = running_corrects.double() / count
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            #scheduler.step()

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        time_elapsed = time.time() - since
        print('Training Epoch complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

file_name = args.data_name+'.pickle'
with open(os.path.join('data',file_name), 'rb') as f:
            dataDict = pickle.load(f) 
        
W2I = make_Vocab(dataDict['test_X'])

train_Xd = [make_Dict(x) for x in dataDict['train_X']]
test_Xd = [make_Dict(x) for x in dataDict['test_X']]

VOCAB_SIZE = min(len(W2I),10000000)
Hasher = FeatureHasher(n_features=VOCAB_SIZE).fit(train_Xd)   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = args.hidden_dim
NUM_CLASS = len(set(dataDict['test_Y']))
NUM_EPOCH = args.num_epoch
LEARN_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
MOMENTUM = args.momentum
VAL_SPLIT = args.val_split


dataloaders= split_Data(TrainDataset(),VAL_SPLIT,BATCH_SIZE)
#test_lodaer  = DataLoader(dataset=TestDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = TextClassifier(VOCAB_SIZE,HIDDEN_DIM,NUM_CLASS).to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARN_RATE, momentum = MOMENTUM)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_model = train_model(model, criterion, optimizer, scheduler, NUM_EPOCH)

model_name = args.data_name+'_model.pth'
torch.save(best_model.state_dict(), os.path.join('model',model_name))


