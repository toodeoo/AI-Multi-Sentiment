import json
import math
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split

def get_text(file, encoding):
    text = ''
    with open(file, encoding=encoding) as fp:
        for line in fp.readlines():
            line = line.strip('\n')
            text += line
    return text

def gec_dataset(values):
    dataset = []
    for i in range(len(values)):
        guid = str(int(values[i][0]))
        tag = values[i][1]
        if type(tag) != str and math.isnan(tag):
            tag = None
        file = path_text + guid + '.txt'
        with open(file, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
            if encoding == "GB2312":
                encoding = "GBK"

        text = ''
        try:
            text = get_text(file, encoding)
        except UnicodeDecodeError:
            try:
                text = get_text(file, 'ANSI')
            except UnicodeDecodeError:
                print('UnicodeDecodeError')
        dataset.append({
            'guid': guid,
            'text': text,
            'label': tag,
            'img': path_img + guid + '.jpg',
        })
    return dataset


path_train = '../train.txt'
path_test = '../test_without_label.txt'
path_text = '../data/'
path_img = './dataset/data/'

train_data_val = pd.read_csv(path_train)
test_data = pd.read_csv(path_test)
train_data, val_data = train_test_split(train_data_val, test_size=0.1, random_state=23)

train_set = gec_dataset(train_data.values)
val_set = gec_dataset(val_data.values)
test_set = gec_dataset(test_data.values)

with open('../train.json', 'w', encoding="utf-8") as f:
    json.dump(train_set, f)

with open('../val.json', 'w', encoding="utf-8") as f:
    json.dump(val_set, f)

with open('../test.json', "w", encoding="utf-8") as f:
    json.dump(test_set, f)