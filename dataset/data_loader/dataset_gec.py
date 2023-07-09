import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import XLMRobertaTokenizer

class MyDataset(Dataset):
    def __init__(self, args, data, transform=None):
        self.args = args
        self.data = data
        self.transform = transform
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained)
        self.label_dict_number = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
        }
        self.label_dict_str = {
            0: 'negative',
            1: 'neutral',
            2: 'positive',
        }

    def __getitem__(self, index):
        return self.tokenize(self.data[index])

    def __len__(self):
        return len(self.data)

    def tokenize(self, item):
        item_id = item['id']
        text = item['text']
        img = item['img']
        label = item['label']
        text_token = self.tokenizer(text, return_tensors="pt", max_length=self.args.text_size, padding='max_length', truncation=True)
        text_token['input_ids'] = text_token['input_ids'].squeeze()
        text_token['attention_mask'] = text_token['attention_mask'].squeeze()

        if self.transform is not None:
            img_token = self.transform(img)
        else:
            img_token = torch.tensor(img)

        if label == 'negative' or label == 'neutral' or label == 'positive':
            label_token = self.label_dict_number[label]
        else:
            label_token = -1

        return item_id, text_token, img_token, label_token


def load_json(file):
    data_list = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = json.load(f)
        for i, line in enumerate(lines):
            item = {}
            img_path = line['img']
            item['img'] = np.array(Image.open(img_path))
            item['text'] = line['text']
            item['label'] = line['label']
            item['id'] = line['guid']
            data_list.append(item)
    return data_list


def load_data(args):
    img_size = (args.img_size, args.img_size)
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size), transforms.Normalize([0.5], [0.5])])
    if args.do_train:
        train_data_list = load_json(args.train_file)
        train_set = MyDataset(args, train_data_list, transform=data_transform)
        val_data_list = load_json(args.val_file)
        val_set = MyDataset(args, val_data_list, transform=data_transform)
        return train_set, val_set
    if args.do_test:
        test_data_list = load_json(args.test_file)
        test_set = MyDataset(args, test_data_list, transform=data_transform)
        val_data_list = load_json(args.val_file)
        val_set = MyDataset(args, val_data_list, transform=data_transform)
        return test_set, val_set


def save_data(file, predict_list):
    with open(file, 'w', encoding='utf-8') as f:
        f.write('guid,tag' + '\n')
        for i in range(len(predict_list)):
            f.write(predict_list[i]['guid'] + ',' + predict_list[i]['tag'] + '\n')
