# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network, predict
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()



if __name__ == '__main__':
    dataset = 'CivilRules'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # predict
    model = x.Model(config).to(config.device)
    labels_pre = predict(config, model, dev_iter)
    print(labels_pre)
    print(labels_pre.shape)

    content ={}
    datapath= r'./'+dataset+'/data/dev.txt'
    with open(datapath, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
        print(len(lines))
        assert len(lines) == len(labels_pre), 'the len of label and predict should be the same'
        texts=''
        title=''
        for index, line in enumerate(lines):
            line=line.rstrip()
            text, label = line.split('\t')
            if label == '1':
                if texts is not '':
                    content[title]=texts
                title = text
                texts = ''
            else:
                texts+=(text+'\t'+label+'\t'+str(labels_pre[index])+'\n')
                if index == len(lines) -1:
                    content[title]=texts

    for key,value in content.items():
        path = r'./'+dataset+'/predict/'+key+'.txt'
        with open(path, 'w', encoding='utf-8') as f1:
            f1.write(value)
