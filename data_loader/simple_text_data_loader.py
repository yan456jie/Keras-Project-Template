from base.base_data_loader import BaseDataLoader
import csv
import random
import jieba  #处理中文
from sklearn.model_selection import train_test_split
from utils.data_clean import clean_build_data,clean_str_comm,clean_calss_data
from utils.data_id_conv import write_word2id,process_file,read_word2id
import os
import numpy

class SimpleTextDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SimpleTextDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.get_search_filter_feature()
        # self.X_train = self.X_train.reshape((-1, 28 * 28))
        # self.X_test = self.X_test.reshape((-1, 28 * 28))
        # x= np.reshape(x, (batch_size , seq_len, input_dim))
        self.X_train = numpy.reshape(self.X_train, (len(self.X_train), self.config.trainer.seq_length, 1))
        self.X_test = numpy.reshape(self.X_test, (len(self.X_test), self.config.trainer.seq_length, 1))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_search_filter_feature(self):
        filter_comment_1 = '/Users/nali/Downloads/filter_comment_1.csv'
        filter_comment_0 = '/Users/nali/Downloads/filter_comment_0.csv'

        filter_comment_1_data = self.read_data_file(filter_comment_1)
        filter_comment_0_data = self.read_data_file(filter_comment_0)

        print('get comment data success!\npos_size = ' + str(len(filter_comment_1_data)) + '\nneg_size = ' + str(len(filter_comment_0_data)))
        # 原始数据（清洗、替换、分词）
        train_list, val_list, test_list, word_list = clean_build_data(filter_comment_1_data, filter_comment_0_data, "true")
        print('\ncomment_train_size = ' + str(len(train_list)) + '\ncomment_val_size = ' + str(
            len(val_list)) + '\ncomment_test_size = '
              + str(len(test_list)) + '\ncomment_word_size = ' + str(len(word_list)))
        # id数据
        labels = ['__label__0', '__label__1']
        label_to_id = dict(zip(labels, range(len(labels))))
        word_to_id = dict(zip(word_list, range(len(word_list))))

        self.config.trainer.embed_feature = len(word_to_id)

        split_punc = '\t|\t'
        write_word2id(self.config.callbacks.model_dir + '/vocab', word_to_id, split_punc)
        write_word2id(self.config.callbacks.model_dir + '/label', label_to_id, split_punc)
        print('\ncomment_word_id size = ' + str(len(word_to_id)) + '\ncomment_label_id_size = ' + str(len(label_to_id)))

        # id表示的数据
        x_train, y_train = process_file(train_list, word_to_id, label_to_id, self.config.trainer.seq_length)
        x_val, y_val = process_file(val_list, word_to_id, label_to_id, self.config.trainer.seq_length)

        return (x_train, y_train), (x_val, y_val)

    def read_data_file(self, path):
        data = []
        with open(path, mode='r', encoding='utf-8') as f:
            f_csv = csv.reader((line.replace('\0', '') for line in f))
            headers = next(f_csv)
            i = 0
            for row in f_csv:
                word = row[0]
                data.append(word)
                i = i+1
                if i>10000:
                    break

        return data


class SimpleTestTextDataLoader(SimpleTextDataLoader):
    def __init__(self, config):
        super(SimpleTextDataLoader, self).__init__(config)
        self.label_to_id = read_word2id(self.config.callbacks.model_dir + '/label', '\t|\t')
        self.word_to_id = read_word2id(self.config.callbacks.model_dir + '/vocab', '\t|\t')

    def get_single_feature(self, text):
        text_list = [text]

        # id表示的数据
        x_train, y_train = self.get_list_feature(text_list)

        return (x_train, y_train)

    def get_list_feature(self, text_list):

        # 原始数据（清洗、替换、分词）
        res_list = clean_calss_data(text_list, '__label__1', 'true')

        # id表示的数据
        x_train, y_train = process_file(res_list, self.word_to_id, self.label_to_id, self.config.trainer.seq_length)
        x_train = numpy.reshape(x_train, (len(x_train), self.config.trainer.seq_length, 1))
        # y_train = numpy.reshape(y_train, (len(y_train), self.config.trainer.seq_length, 1))
        return (x_train, y_train)


class SimpleCnnTextDataLoader(SimpleTextDataLoader):
    def __init__(self, config):
        super(SimpleTextDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.get_search_filter_feature()
        # x= np.reshape(x, (batch_size , seq_len, input_dim))
        self.X_train = numpy.reshape(self.X_train, (len(self.X_train), self.config.trainer.seq_length))
        self.X_test = numpy.reshape(self.X_test, (len(self.X_test), self.config.trainer.seq_length))