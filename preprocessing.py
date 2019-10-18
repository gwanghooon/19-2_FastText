import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pickle

nltk.download('stopwords')
nltk.download('wordnet')

data_path = 'data/ag_news_csv/'

train_data = pd.read_csv(data_path+'train.csv',names = ['class','title','text'])
test_data = pd.read_csv(data_path+'test.csv',names = ['class','title','text'])
train_data = train_data.sample(frac=1).reset_index(drop=True)
test_data = test_data.sample(frac=1).reset_index(drop=True)

def preprocessing(text):
    
    text = text.replace('\\',' ')
    
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

print('Preprocessing....')

for i, row in train_data.iterrows():
    
    tokens = preprocessing(row['title']) + bigram(preprocessing(row['title'])) + preprocessing(row['text']) + bigram(preprocessing(row['text']))
    train_X.append(tokens)
    
    cls = row['class']-1
    train_Y.append(cls)

for i, row in test_data.iterrows():
    
    tokens = preprocessing(row['title']) + bigram(preprocessing(row['title'])) + preprocessing(row['text']) + bigram(preprocessing(row['text']))
    test_X.append(tokens)
    
    cls = row['class']-1
    test_Y.append(cls)

print('Preprocessing Done!')

dataDict = {'train_X':train_X,'train_Y':train_Y,'test_X':test_X,'test_Y':train_Y}

pickle.dump(dataDict , open('data/ag_news.pickle','wb'))
