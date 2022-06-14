import numpy as np
import h5py
import json
import os


def transfer_datas(path, name, labels_2_ids, feature_dic, global_corpus, max_length):
    """

    :param path: 数据存储地址
    :param name: h5名称
    :param labels_2_ids:标签和数值的转换字典
    :param feature_dic: 特征字典
    :param global_corpus: 全局数据
    :param max_length:最大长度
    :return:经过该数据添加过的转换字典
    """
    lengths = []
    num = len(global_corpus)
    labels_out = np.zeros((num, 1))
    feature_out = np.zeros((num, max_length, 312))

    for i, entity_sentence in enumerate(global_corpus):

        sen_id = entity_sentence['id']
        input_offset = entity_sentence['input_offset']
        label_content = json.loads(entity_sentence['label_content'])
        sentence = label_content['content']
        label = label_content['label']
        if label not in labels_2_ids.keys():
            labels_2_ids[label] = len(labels_2_ids.keys())
        labels_out[i, 0] = labels_2_ids[label]
        sentence_length = len(sentence)
        # print(sentence,feature_dic[sen_id]['content'][input_offset:input_offset + sentence_length])
        assert len(sentence) == len(feature_dic[sen_id]['content'][input_offset:input_offset + sentence_length])

        if sentence_length < max_length:
            feature_out[i, :sentence_length, :] = feature_dic[sen_id]['feature'][:,
                                                  input_offset:input_offset + sentence_length, :]
            lengths.append(sentence_length)
        else:
            feature_out[i, :max_length, :] = feature_dic[sen_id]['feature'][:,
                                             input_offset:input_offset + max_length, :]
            lengths.append(max_length)

        if os.path.exists(path + name + ".h5"):
            os.remove(path + name + ".h5")
        with h5py.File(path + name + ".h5", "w") as f:
            out_datas = f.create_group('classification_global_datas')
            out_datas['features'] = feature_out
            out_datas['labels'] = labels_out
            out_datas['lengths'] = lengths
    return labels_2_ids


def save_features(data_path, datas, max_length, kfold=5):
    """

    :param data_path: 数据存储路径
    :param datas: 数据集和特征集
    :param max_length: 最大长度
    :param kfold: 交叉验证数量
    :return:
    """
    global_corpus = datas['global_corpus']
    feature_corpus = datas['feature_corpus']
    if len(global_corpus) == 1:
        global_corpus.append(global_corpus[0])
    np.random.shuffle(global_corpus)
    feature_dic = {}
    for feature in feature_corpus:
        feature_dic[feature['id']] = {
            'content': feature['content'],
            'feature': np.array(feature['feature'])
        }
    labels_2_ids = {}

    # if len(global_corpus) >= 2000:
    #     global_corpus = global_corpus[:2000]

    start = 0
    for i in range(kfold):
        end = int(len(global_corpus) / kfold * (i + 1))
        one_fold_global_corpus = global_corpus[start:end]
        start = end
        name = 'fold_' + str(i).zfill(2)
        labels_2_ids = transfer_datas(data_path, name, labels_2_ids, feature_dic, one_fold_global_corpus, max_length)

    with open(data_path + 'label_2_ids.json', 'w', encoding='utf-8')as f:
        f.write(json.dumps(labels_2_ids, indent=4, ensure_ascii=False))

    return len(labels_2_ids.keys())
