import os
import argparse
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pickle
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type = str, help = 'dataset name')

args = parser.parse_args()

# nltk.download('stopwords')
# nltk.download('wordnet')

data_path = os.path.join('data',args.data_name+'_csv')

train_data = pd.read_csv(os.path.join(data_path,'train.csv'),names = ['class','title','text'])
test_data = pd.read_csv(os.path.join(data_path,'test.csv'),names = ['class','title','text'])
train_data = train_data.sample(frac=1).reset_index(drop=True)
test_data = test_data.sample(frac=1).reset_index(drop=True)

def preprocessing(text):
    
    text = str(text).replace('\\',' ')
    
    tokens = [word for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]
    
    stop = stopwords.words('english')
    
    tokens = [token for token in tokens if token not in stop]

    tokens = [word for word in tokens if len(word) >= 3]

    tokens = [word.lower() for word in tokens]

    # lemmatization
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]

    tokens = [lmtzr.lemmatize(word, 'v') for word in tokens]

    # stemming
    stemmer = PorterStemmer()
    tokens = [ stemmer.stem(word) for word in tokens ]

    #preprocessed_text= ' '.join(tokens)
    
    return tokens

def bigram(tokens):
    
    bi = []
    
    for i,t in enumerate(tokens[:-1]):
        bi.append(tokens[i]+' '+tokens[i+1])
        
    return bi

train_X,train_Y = [],[] 
test_X,test_Y = [],[] 

print('# of train:',len(train_data))
print('# of test:',len(test_data))


print('Preprocessing....')
since = time.time()



for i, row in tqdm(train_data.iterrows()):
    
    tokens = preprocessing(row['title']) + bigram(preprocessing(row['title'])) + preprocessing(row['text']) + bigram(preprocessing(row['text']))
    train_X.append(tokens)
    
    cls = row['class']-1
    train_Y.append(cls)

for i, row in tqdm(test_data.iterrows()):
    
    tokens = preprocessing(row['title']) + bigram(preprocessing(row['title'])) + preprocessing(row['text']) + bigram(preprocessing(row['text']))
    test_X.append(tokens)
    
    cls = row['class']-1
    test_Y.append(cls)

print('Preprocessing Done!')
time_elapsed = time.time() - since
print('Preprocessing complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

dataDict = {'train_X':train_X,'train_Y':train_Y,'test_X':test_X,'test_Y':train_Y}
                         
file_name = args.data_name+'.pickle'
pickle.dump(dataDict , open(os.path.join('data',file_name),'wb'))
