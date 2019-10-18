import pickle
from sklearn.feature_extraction import FeatureHasher
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


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


with open('data/ag_news.pickle', 'rb') as f:
            dataDict = pickle.load(f) 
        
W2I = make_Vocab(dataDict['test_X'])

train_Xd = [make_Dict(x) for x in dataDict['train_X']]
test_Xd = [make_Dict(x) for x in dataDict['test_X']]

VOCAB_SIZE = min(len(W2I),10000000)
Hasher = FeatureHasher(n_features=VOCAB_SIZE).fit(train_Xd)   

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