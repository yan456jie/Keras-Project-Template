# -*- coding: utf-8 -*-
import jieba
import random
import re
from idna import unichr
import codecs

# base_dir = './Keras-Project-Template/'
base_dir = '../'

def read_stop_word(stop_path):
    stop_set = set()
    # with open(stop_path, 'r') as stops:
    with codecs.open(stop_path, 'r', encoding='utf-8') as stops:
        for line in stops:
            stop_set.add(line.strip().lower())
    return stop_set

stop_set = read_stop_word(base_dir + 'data/dict/stop_words.txt')
dict_set = read_stop_word(base_dir + 'data/dict/dict.txt')

# 专辑名数据清洗
def clean_str_title(string, stop_set, dict_set):

    string = quanToBan(string).strip().lower()

    string = re.sub(re.compile(r"\[[\u4E00-\u9FA5a-z]{1,4}\]"), "", string)
    string = re.sub(re.compile(r"[\t\r\n]+"), "", string)
    string = re.sub(re.compile(r"[^\u4E00-\u9FA5A-Za-z0-9]+"), "", string)
    # 处理手机号码
    string = re.sub(re.compile(r"(\\+86)?([₁¹①1一壹幺][₃₅₆₇₈₉³⁵⁶⁷⁸⁹③⑤⑥⑦⑧⑨356789三叁五伍六陆七柒八捌九玖]" +
                                                         "[₀₁₂₃₄₅₆₇₈₉¹²³⁴⁵⁶⁷⁸⁹⁰⓪①②④③⑤⑥⑦⑧⑨0-9一壹二贰貮三叁四肆五伍六陆七柒八捌九玖零幺o〇]{9})"), " #cellphone ", string)
    # 处理通用数字
    for num in re.findall(re.compile(r"[\d₀₁₂₃₄₅₆₇₈₉¹²³⁴⁵⁶⁷⁸⁹⁰⓪①②④③⑤⑥⑦⑧⑨一壹二贰貮三叁四肆五伍六陆七柒八捌九玖零幺〇]{3,20}"), string):
        string = string.replace(num, " #num" + str(len(num)) + " ", 1)
    # 处理无序字母
    for part in re.findall(re.compile(r"[a-zA-Z0-9]+"), string):
        if part not in dict_set and len(part) >= 3:
            string = string.replace(part, " #alpha"+str(len(part))+" ", 1)

    wordList = []
    for part in string.split(" "):
        if not re.search(re.compile(r"[\u4E00-\u9FA5]"), part) and part not in stop_set:
            wordList.append(str(part).lower().strip())
        else:
            for word in list(jieba.cut(part)):
                if word not in stop_set:
                    # 分词
                    wordList.append(str(word).lower().strip())
    return wordList


# 评论数据清洗
def clean_str_comm(string, stop_set):
    string = string.strip().lower()
    string = re.sub(re.compile(r"\[[\u4E00-\u9FA5a-z]{1,4}\]"), " ", string)
    filtrate = re.compile(u'[^\u4E00-\u9FA5A-Za-z0-9_]+')  # 中文字,字母,下划线
    string = filtrate.sub(r' ', string)  # replace
    string = string.replace("[\t\r\n ]+", " ")
    seg_content = list([word for word in jieba.cut(string.strip()) if word not in stop_set and len(word) < 10])
    return seg_content

# 清洗某类数据
def clean_calss_data(ori_list, label, stop_set, dict_set, is_comment):
    res_list = []
    for sen in ori_list:
        if not is_comment:
            word_list = clean_str_title(sen,stop_set, dict_set)
        else:
            word_list = clean_str_comm(sen, stop_set)
        word_list.insert(0,label)
        if len(word_list) > 1:
            res_list.append(word_list)
    return res_list

# 数据清洗接口
def clean_build_data(pos_data, neg_data, is_comment):

    data_pos_list = clean_calss_data(pos_data, '__label__1', stop_set, dict_set, is_comment)
    data_neg_list = clean_calss_data(neg_data, '__label__0', stop_set, dict_set, is_comment)

    data_pos_list.extend(data_neg_list)
    train_list, test_list, word_set = [], [],set()

    # 数据修正/构造词典
    for word_list in data_pos_list:
        if '#cellphone' in word_list:
            word_list[0] = '__label__1'
        for word in word_list[1:]:
            if len(word) > 0:
                word_set.add(word)
    # 打乱数据
    random.shuffle(data_pos_list)

    # 数据切分
    length = len(data_pos_list)
    test_list = data_pos_list[int(0.95*length):length]
    val_list = data_pos_list[int(0.9*length):int(0.95*length)]
    train_list = data_pos_list[0:int(length*0.9)]
    word_list = list(word_set)
    word_list.insert(0, '<PAD>')

    return train_list, val_list, test_list, word_list

def quanToBan(quan_String):
    # 全角转半角
    res_String = ""
    for uchar in quan_String:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        res_String += unichr(inside_code)
    return res_String

