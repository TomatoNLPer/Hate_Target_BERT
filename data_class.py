import torch
from torch.utils.data import Dataset
import os
from transformers import BertTokenizer
import torchvision
import csv
import json


class Sentiment(Dataset):
    def __init__(self, file_name):
        super(Sentiment, self).__init__()
        self.root_path = os.path.join('./data/', file_name)
        # self.image_path = os.path.join(self.root_path,'images')
        # self.data_split = data_split
        self.max_txt_length = 128
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        self.data_base = self.load_annotations()
        self.file_name = file_name
#
    def load_annotations(self):
        anno_path = self.root_path
        #valid_json_path = './data/HarmC/annotations/val.jsonl'
        file = open(anno_path,'r',encoding='utf-8')
        reader = csv.DictReader(file)
        text_column = [row for row in reader]
        return text_column


    def __len__(self):
        return len(self.data_base)


    def __getitem__(self, index):
        anno = self.data_base[index]
        label = int(anno['label'])
        input_text_features = self.tokenizer.encode_plus(
            anno['sentence'],
            max_length=self.max_txt_length,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        # text_features = self.tokenizer(anno['text'], max_length=100, truncation=True, padding=True)
        return {'input_ids':input_text_features['input_ids'].squeeze(dim=0),\
                'token_type_ids':input_text_features['token_type_ids'].squeeze(dim=0),\
                'attention_mask':input_text_features['attention_mask'].squeeze(dim=0),\
                'label':label
                }

class Target(Dataset):
    def __init__(self, file_name):
        super(Target, self).__init__()
        self.root_path = file_name
        # self.image_path = os.path.join(self.root_path,'images')
        # self.data_split = data_split
        self.max_txt_length = 128
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        self.data_base = self.load_annotations()
        # self.file_name = file_name
#
    def load_annotations(self):
        anno_path = self.root_path
        #valid_json_path = './data/HarmC/annotations/val.jsonl'
        file = open(anno_path,'r',encoding='utf-8')
        reader = csv.DictReader(file)
        text_column = [row for row in reader]
        return text_column


    def __len__(self):
        return len(self.data_base)


    def __getitem__(self, index):
        anno = self.data_base[index]
        # label = int(anno['label'])
        input_text_features = self.tokenizer.encode_plus(
            anno['sentence'],
            max_length=self.max_txt_length,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        # text_features = self.tokenizer(anno['text'], max_length=100, truncation=True, padding=True)
        return {'input_ids':input_text_features['input_ids'].squeeze(dim=0),\
                'token_type_ids':input_text_features['token_type_ids'].squeeze(dim=0),\
                'attention_mask':input_text_features['attention_mask'].squeeze(dim=0),\
                'raw_sentence': anno['sentence']
                }

class Pair_Target(Dataset):
    def __init__(self, file_name):
        super(Pair_Target, self).__init__()
        # self.root_path = os.path.join('./data/', file_name)
        # self.image_path = os.path.join(self.root_path,'images')
        # self.data_split = data_split
        self.max_txt_length = 128
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        self.prompt_list, self.gold_list = self.load_prompt_gold(file_name)
        self.file_name = file_name
#
    def load_annotations(self):
        anno_path = self.root_path
        #valid_json_path = './data/HarmC/annotations/val.jsonl'
        file = open(anno_path,'r',encoding='utf-8')
        reader = csv.DictReader(file)
        text_column = [row for row in reader]
        return text_column

    def load_prompt_gold(self,in_f):
        gold_list, pred_list = [], []
        file = open(in_f, 'r', encoding='utf-8')
        result_dict = json.load(file)
        file.close()
        for key in result_dict.keys():
            sample = result_dict[key]
            gold_list.append(sample['hs'])
            pred_list.append(sample['cs'])
        return pred_list[:2000],gold_list[:2000]


    def __len__(self):
        return len(self.prompt_list)


    def __getitem__(self, index):

        gold_sentence = self.gold_list[index]
        predict_sentence = self.prompt_list[index]
        # label = int(anno['label'])
        input_gold_features = self.tokenizer.encode_plus(
            gold_sentence,
            max_length=self.max_txt_length,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        input_predict_features = self.tokenizer.encode_plus(
            predict_sentence,
            max_length=self.max_txt_length,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        # text_features = self.tokenizer(anno['text'], max_length=100, truncation=True, padding=True)
        return {'gold_input_ids':input_gold_features['input_ids'].squeeze(dim=0),\
                'gold_token_type_ids':input_gold_features['token_type_ids'].squeeze(dim=0),\
                'gold_attention_mask':input_gold_features['attention_mask'].squeeze(dim=0),\
                'predict_input_ids': input_predict_features['input_ids'].squeeze(dim=0),\
                'predict_token_type_ids': input_predict_features['token_type_ids'].squeeze(dim=0),\
                'predict_attention_mask': input_predict_features['attention_mask'].squeeze(dim=0),\
                }