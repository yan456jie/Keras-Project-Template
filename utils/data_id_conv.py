# -*- coding: utf-8 -*-
import numpy as np
import keras as kr
import codecs

def split_line(train_list):
    data_train, labels = [],[]
    for sen in train_list:
        if sen[0] in ['__label__0','__label__1']:
            data_train.append(sen[1:])
            labels.append(sen[0])
    return data_train, labels

def process_file(test_list, word_to_id, cat_to_id, max_length=500):
    """将文件转换为id表示"""
    contents, labels = split_line(test_list)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def write_word2id(path, word2id, split_punc):
    # with open(path, 'w') as f:
    with codecs.open(path, 'w', encoding='utf-8') as f:
        for word,id in word2id.items():
            f.write(word + split_punc + str(id) + '\n')

def read_word2id(vocab_path, split_punc):
    word_to_id = {}
    # with open(vocab_path, 'r') as f:
    with codecs.open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                word_to_id[line.strip().split(split_punc)[0]] = int(line.strip().split('\t|\t')[1])
            except:
                continue
    return word_to_id
