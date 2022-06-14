import numpy as np
import h5py
import json
import os
from structure.bert import fast_bert_train
import re


def get_senstence_label(all_datas):
    """

    :param all_datas:数据集
    :return: 句子与标签
    """
    sentences = []
    labels = []
    for data in all_datas:
        contexts = data
        sentence = ''
        label = []
        for context in contexts:
            if context[0] != ' ':
                sentence = sentence + context[0]
                label.append(context[1])
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels


def get_label_ids(path, labels):
    """

    :param path:数据存储路径
    :param labels: 所有的标签
    :return:标签映射字典
    """
    labels_2_ids = {'O': 0}
    for data in labels:
        for label in data:
            if label not in labels_2_ids.keys():
                labels_2_ids[label] = len(labels_2_ids.keys())
    with open(path + 'label_2_ids.json', 'w', encoding='utf-8')as f:
        f.write(json.dumps(labels_2_ids, indent=4, ensure_ascii=False))
    return labels_2_ids


def label_2_ids(labels, labels_2_ids):
    """

    :param labels: 所有的标签
    :param labels_2_ids: 标签映射字典
    :return: 转换之后的标签
    """
    outs = []
    for label in labels:
        out = []
        for single_label in label:
            out.append(labels_2_ids[single_label])
        outs.append(out)
    return outs


def split_punctuate_datas(datas):
    """

    :param datas:所有的数据
    :return: 将数据中的标点提取出来，生成相应的标签
    """
    new_datas = []
    for data in datas:
        new_data = []
        data = re.sub(r'\r|\t|\n| ', '$', data['sentence']['context'])
        for word in data:
            if word in ['，', '。', '；', ';']:
                if word not in ['；', ';']:
                    new_data.append(['@', 'B-' + word])
                else:
                    new_data.append(['@', 'B-；'])
            else:
                new_data.append([word, 'O'])
        new_datas.append(new_data)
    return new_datas


def kfold_transfer_punctuate_datas_labels(path, datas, kfold=5):
    """

    :param path: 数据存储路径
    :param datas: 所有数据
    :param kfold: 交叉验证的数量
    :return: 分割之后的数据集
    """
    np.random.shuffle(datas)
    datas = split_punctuate_datas(datas)
    datas, labels = get_senstence_label(datas)
    labels_2_ids = get_label_ids(path, labels)
    labels = label_2_ids(labels, labels_2_ids)

    start = 0
    end = len(datas)
    folds_datas_labels = []

    for i in range(kfold):
        end = int(len(datas) / float(kfold) * (i + 1))
        if end > len(datas):
            end = len(datas)
        one_fold = {'datas': datas[start:end], 'labels': labels[start:end]}
        start = end
        folds_datas_labels.append(one_fold)
    return folds_datas_labels


def num_to_char(num):
    num = int(num.group(0))
    num = str(num)
    num_dict = {"0": u"零", "1": u"一", "2": u"二", "3": u"三", "4": u"四", "5": u"五", "6": u"六", "7": u"七", "8": u"八",
                "9": u"九"}
    listnum = list(num)
    shu = []
    for i in listnum:
        shu.append(num_dict[i])
    new_str = "".join(shu)
    return new_str


def process_str(data_ch):
    num_cpl = re.compile(r'\d')
    data_ch = num_cpl.sub(num_to_char, data_ch)
    return data_ch


def get_ner_features(path, name, datas, labels, max_length):
    """

    :param path:数据存储的路径
    :param name:h5的名字
    :param datas:所有数据
    :param labels:所有标签
    :param max_length:句子最大长度
    :return:
    """
    token_results = []
    features_out = np.zeros([len(datas), max_length, 312], dtype=np.float32)
    for num, data in enumerate(datas):
        data = process_str(data)
        data = list(data)[:max_length]
        feature, token = fast_bert_train.predict([data])
        features_out[num, :len(data), :] = feature[0, 1:len(data) + 1, :]
        token_results.append(token[0][1:len(data) + 1])

    labels_out = np.zeros((len(datas), max_length))
    real_labels = []
    lengths = []

    for i, tokens in enumerate(token_results):
        lengths.append(len(tokens))
        for j, token in enumerate(tokens):
            labels_out[i, j] = labels[i][j]
            real_labels.append(labels[i][j])

    if os.path.exists(path + name + ".h5"):
        os.remove(path + name + ".h5")
    with h5py.File(path + name + ".h5", "w") as f:
        out_datas = f.create_group('punctuate_datas')
        out_datas['features'] = features_out
        out_datas['labels'] = labels_out
        out_datas['real_labels'] = real_labels
        out_datas['lengths'] = lengths


def save_features(data_path, datas, max_length, kfold=5):
    """

    :param data_path:数据存储路径
    :param datas: 数据
    :param max_length:句子最大长度
    :param kfold: 交叉验证的数量
    :return:
    """
    folds_datas_labels = kfold_transfer_punctuate_datas_labels(data_path, datas, kfold)
    for i, fold in enumerate(folds_datas_labels):
        datas = fold['datas']
        labels = fold['labels']
        name = 'fold_' + str(i).zfill(2)
        get_ner_features(data_path, name, datas, labels, max_length)
