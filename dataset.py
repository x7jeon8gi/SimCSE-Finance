
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm
from typing import List, Tuple, Union, Dict
import datasets
from transformers import AutoTokenizer
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

class UnsupDataset(Dataset):
    
    def __init__(self, mode, config):
        super().__init__()
        
        self.data_full_path = glob( os.path.join(os.getcwd(), 'NLP_STUDY', config['data']["data_full_path"]))
        self.tokenizer = AutoTokenizer.from_pretrained(config['transformer']["from_pretrained"])
        self.split_day = config['data']['split_day']
        self.max_length = config['transformer']['max_length']
        
        if mode == "train":
            self.data_path = [path for path in self.data_full_path if int(path[-12:-4]) < self.split_day]       
        
        elif mode == "test":
            self.data_path = [path for path in self.data_full_path if int(path[-12:-4]) >= self.split_day]

        #todo: File.lock() <- prevent from reading the same file
        self.dataset_collector = self.load_datasets()
        self.samples = self.process_datasets()
        
    def load_datasets(self):
        #todo: parallel with concurrent.futures
        
        dataset_collector = []
        
        print("Load datasets...")
        for path in tqdm(self.data_path):
            file = pd.read_pickle(path)

            text = pd.DataFrame(file['텍스트'])
            dataset_collector.append(text)
            
        # def load_pickle_file(path):
        #     file = pd.read_pickle(path)
        #     text = pd.DataFrame(file['텍스트'])
        #     return text
        
        # with ProcessPoolExecutor(max_workers=4) as executor:
        #     for text in tqdm(executor.map(load_pickle_file, self.data_path), total=len(self.data_path)):
        #         dataset_collector.append(text)
        
        return dataset_collector
    
    def process_datasets(self):
        #todo: parallel with huggingface datasets map -> complete 
        # samples = []
        
        # for dataset in tqdm(self.dataset_collector):
        #     dataset = [self.tokenizer.encode(text) for text in dataset]
        #     samples.append(dataset)
        
        # hf_datasets = {key: datasets.Dataset.from_pandas(df) for key, df in enumerate(self.dataset_collector)}
        # for idx, hf_data in tqdm(hf_datasets.items()):
        #     hf_data = hf_data.map(lambda x: self.tokenizer(x['텍스트'], max_lenght=512, truncation=True), batched=True, remove_columns=['텍스트']).with_format("torch", columns = ['input_ids', 'attention_mask'])
        #     samples.extend(hf_data)
        # return samples    
        
        hf_datasets = pd.concat(self.dataset_collector, ignore_index=True)
        hf_datasets = datasets.Dataset.from_pandas(hf_datasets)
        hf_datasets = hf_datasets.map(lambda x: self.tokenizer(x['텍스트'], max_length=self.max_length, truncation=True, padding="max_length"), batched=True, remove_columns=['텍스트']).with_format("torch", columns = ['input_ids', 'token_type_ids', 'attention_mask'])
            
        return hf_datasets
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    
class SupervisedDataset(Dataset):
        
        def __init__(self, mode, config):
            super().__init__()
            
            self.data_full_path = glob( os.path.join(os.getcwd(), config['data']["data_full_path"]))
            self.tokenizer = AutoTokenizer.from_pretrained(config['transformer']["from_pretrained"])
            self.split_day = config['data']['split_day']
            self.max_length = config['transformer']['max_length']
            
            if mode == "train":
                self.data_path = [path for path in self.data_full_path if int(path[-12:-4]) < self.split_day]       
            
            elif mode == "test":
                self.data_path = [path for path in self.data_full_path if int(path[-12:-4]) >= self.split_day]
            
            self.dataset_collector = self.load_datasets()
            self.samples = self.process_datasets()
            
        def load_datasets(self):
            
            dataset_collector = []
            
            print("Load Supervised datasets..!")
            for path in tqdm(self.data_path):
                file = pd.read_pickle(path)

                anchor = file['anchor']
                positive = file['positive']
                negative = file['negative']
                
                dataset_collector.append([anchor, positive, negative])
                
            dataset_collector = pd.DataFrame(dataset_collector, columns=['anchor', 'positive', 'negative'])
            return dataset_collector
        
        def process_datasets(self):
            #! we do not use tokenizer here b.c. memory issue
            hf_datasets = datasets.Dataset.from_pandas(self.dataset_collector)
            assert len(hf_datasets['anchor']) == len(hf_datasets['positive']) == len(hf_datasets['negative'])
            return hf_datasets
        
        def convert_to_tensor(self, text):
            #* It returns a dictionary of tensors
            return torch.tensor(self.tokenizer(text, max_length=self.max_length, truncation=True, padding="max_length"), return_tensors='pt')
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            anchor = self.convert_to_tensor(self.samples['anchor'][idx])
            positive = self.convert_to_tensor(self.samples['positive'][idx])
            negative = self.convert_to_tensor(self.samples['negative'][idx])
            
            tensors = {
                'anchor': anchor,
                'positive': positive,
                'negative': negative
            }
            
            return tensors