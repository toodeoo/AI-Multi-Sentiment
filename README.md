# AI-Multi-Sentiment
This is the official repository of DaSE 2023 AI project Multimodal Sentiment Analysis.

## Setup
This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.12.0

- torchvision==0.13.0

- transformers==4.20.1

- numpy==1.22.3

- sklearn==1.1.1

- pandas==1.4.2

- chardet==4.0.0

- Pillow==9.2.0

- tqdm==4.64.0

You can simply run
```shell
pip install -r requirements.txt
```

## Repository structure

We select some important files for detailed description.
```
|-- dataset/
    |-- data/ # the original data
    |-- train.txt # the original data with label
    |-- test_without_label.txt # the original test data without label
    |-- train.json # the processed train set
    |-- test.json # the processed test set
    |-- val.json # the processed val set
|-- model/
    |-- muti_model.py # the implemented model
    |-- baseline.py # includes all base-model implementations
|-- README.md
|-- main.py # the main code
|-- requirements.txt # dependencies
```
## Train and Test
You can simply try to train our model by the script (using default arguments):
```shell
python main.py --do_train
```
You can simply try to test our model by the script, if you have simply tried in Train stage (using default arguments):
```shell
python main.py --do_test
```
## Results
The results are shown in this Table(Accuracy):
| fuse-Strategy             | Only Text | Only Image | Multi-modal |
| ------------------------- | --------- | ---------- | ----------- |
| Concatenate               | 0.6975    | 0.6725     | 0.7375      |
| Multi-head Self-attention | 0.635     | 0.68       | 0.695       |
| Transformer Encoder       | 0.6675    | 0.68       | 0.7         |

## Reference

`https://github.com/RecklessRonan/GloGNN/blob/master/readme.md`
