import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import os
import re
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import pandas as pd
import csv

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class IMDB(Dataset):
    def __init__(self, type, max_len =500):
        self.data = []
        data = pd.read_csv(f'{type}.csv')
        reviews = data['review'].tolist()
        ids = data['ID'].tolist()
        if type == 'train':
            sentiments = data['sentiment'].tolist()
        else:
            sentiments = np.zeros(len(reviews))
        reviews, max_len = self.get_token2num_maxlen(reviews)
        max_len = 500
        
        for id, review, sentiment in zip(ids, reviews,sentiments):
            #zero padding
            if max_len > len(review):
                padding_cnt = max_len - len(review)
                review += padding_cnt * [0]
            else:
                review = review[:max_len]
            #label
            label = sentiment

            self.data.append([id, review, label])

    def __getitem__(self,index):
        ids = torch.tensor(self.data[index][0])
        datas = torch.tensor(self.data[index][1])
        labels = torch.tensor(self.data[index][2])
        
        return ids, datas, labels
    
    def __len__(self):
    
        return len(self.data)
        
    def preprocess_text(self,sentence):
        #delete html tag
        sentence = re.sub(r'<[^>]+>',' ',sentence)
        #delete numbers and punctuations
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        #delete single word
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        #delete spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        #delete spaces of head and tail
        sentence= sentence.strip()
        #captital to lower
        sentence = sentence.lower()
    
        return sentence
    
    
    def get_token2num_maxlen(self, reviews, enable=True):
        token = []
        for review in reviews:
            #split data and store it in list
            review = self.preprocess_text(review)
            #利用set()回傳一個聯集，並且通過迴圈創建一個文字對應的dict方便轉換
            #這邊要注意開頭是1，0通常會作為padding token
            token += review.split(' ')
        
        token_to_num = {data:cnt for cnt,data in enumerate(list(set(token)),1)}
         
        num = []
        max_len = 0 
        for review in reviews:
            review = self.preprocess_text(review)
            tmp = []
            for token in review.split(' '):
                #word to vector
                tmp.append(token_to_num[token])
            #find max 
            if len(tmp) > max_len:
                max_len = len(tmp)
            num.append(tmp)
            
                
        return num, max_len
#find path of current program
py_path = os.path.abspath(__file__)
#change path to current program
py_dir = os.path.dirname(py_path)
os.chdir(py_dir)
#
train_dataset = IMDB(os.path.join(py_dir,'train'))
test_dataset = IMDB(os.path.join(py_dir,'test'))
with open('submission.csv', 'w', newline='') as csvfile:
    fieldnames = ['ID', 'sentiment']
    # put dictionary into CSV file
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # write header
    writer.writeheader()
    writer = csv.writer(csvfile)
    # put result into array
    for id,  reviews, sentiment in test_dataset:
        writer.writerow([int(id.numpy()), int(sentiment.numpy())])