from structure.bert import fast_bert_train
import numpy as np
import h5py
import os
import json
import re


def get_labels(path, datas):
    """

    :param path: 标签转换地址的存储位置
    :param datas: 训练数据
    :return:
    """
    labels = {}
    for data in datas:
        if data['sentence']["context"][1] not in labels.keys():
            labels[data['sentence']["context"][1]] = len(labels.keys())
    with open(path + 'label_2_ids.json', 'w', encoding='utf-8')as f:
        f.write(json.dumps(labels, indent=4, ensure_ascii=False))
    return labels


def kfold_transfer_classification_datas_labels(path, datas, kfold):
    """

    :param path: 标签转换文件的存储地址
    :param datas: 训练数据
    :param kfold: 交叉验证的数量
    :return:
    """
    labels_2_ids = get_labels(path, datas)
    np.random.shuffle(datas)
    start = 0
    end = len(datas)
    folds_datas_labels = []

    for i in range(kfold):
        one_fold = {}
        sentences = []
        labels = []
        end = int(len(datas) / kfold * (i + 1))
        if end > len(datas):
            end = len(datas)
        datas_one_fold = datas[start:end]
        start = end
        for data in datas_one_fold:
            sentences.append(data['sentence']['context'][0])
            labels.append(labels_2_ids[data['sentence']['context'][1]])

        one_fold = {'sentences': sentences, 'labels': labels}
        folds_datas_labels.append(one_fold)
    return folds_datas_labels, len(labels_2_ids.keys())


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


def get_classification_features(path, name, datas, labels):
    """

    :param path: h5文件存储路径
    :param name: 数据集的名字
    :param datas: 数据
    :param labels: 对标签进行转换的字典
    :return:
    """
    labels_out = np.zeros([len(datas), 1], dtype=np.int32)
    feature_results = np.zeros([len(datas), 1, 312], dtype=np.float32)
    for num, data in enumerate(datas):
        data = re.sub(r'\r|\t|\n| ', '$', data)
        data = process_str(data)
        feature, _ = fast_bert_train.predict([list(data)])
        feature_results[num, 0, :] = feature[0, 0, :]
        labels_out[num, 0] = labels[num]

    if os.path.exists(path + name + ".h5"):
        os.remove(path + name + ".h5")

    with h5py.File(path + name + ".h5", "w") as f:
        out_datas = f.create_group('classification_data')
        out_datas['features'] = feature_results[:, 0, :]
        out_datas['labels'] = labels_out


def save_features(data_path, datas, kfold=5):
    """

    :param data_path: 数据存储路径
    :param datas: 训练数据
    :param kfold: 交叉验证的数量
    :return:
    """
    folds_datas_labels, label_nums = kfold_transfer_classification_datas_labels(data_path, datas, kfold)
    for i, fold in enumerate(folds_datas_labels):
        datas = fold['sentences']
        labels = fold['labels']
        name = 'fold_' + str(i).zfill(2)
        get_classification_features(data_path, name, datas, labels)
    return label_nums
