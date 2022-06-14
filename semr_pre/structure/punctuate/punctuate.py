import tensorflow as tf
from tensorflow.python.platform import gfile
from structure.bert import fast_bert, features_dict
import json
import re
import os
import numpy as np
from log import logger

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


def get_punctuations(results):
    """
    标点修正和特征修正
    :param results:将预测之后的实体识别输入，转换成标点修正结果，并将特征进行替换
    :return: 标点修正结果和修正后的特征
    """
    sentence_id = results['id']
    feature = results['feature']
    sentence = results['sentence']
    predict_results = results['result']
    result_sentence = ''
    for i, result in enumerate(predict_results):
        if result != 'O' and sentence[i] in [',', '，', '。', '；']:
            result_sentence += result.split('-')[-1]
            feature[:, i, :] = features_dict[result.split('-')[-1]]['feature']
        else:
            result_sentence += sentence[i]

    return {'id': sentence_id, 'result': result_sentence, 'context': sentence}, feature


def get_features(data):
    """
    获取句子特征
    :param data:句子
    :return:特征
    """
    input_list = []
    data = re.sub(r'\r|\t|\n| ', '$', data)
    data = process_str(data)
    for i, word in enumerate(list(data)):
        if word in [',', '，', '。', '；']:
            input_list.append('@')
        else:
            input_list.append(word)
    if len(data) <= 500:
        feature, token = fast_bert.predict([input_list])
        result_token = token[0]
        result_feature = feature[:, :len(token[0]), :]
    else:
        result_feature = np.zeros([1, len(data) + 2, 312], dtype=np.float32)
        result_token = []
        num2 = int(len(input_list) / 500)
        for i in range(num2 + 1):
            if i != num2:
                feature, token = fast_bert.predict([input_list[i * 500:(i + 1) * 500]])
                feature = feature[:, :len(token[0]), :]
                if i == 0:
                    result_token.extend(token[0][:-1])
                    result_feature[:, i * 500:(i + 1) * 500 + 1, :] = feature[:, :-1, :]
                else:
                    result_token.extend(token[0][1:-1])
                    result_feature[:, i * 500 + 1:(i + 1) * 500 + 1, :] = feature[:, 1:-1, :]
            else:
                feature, token = fast_bert.predict([input_list[i * 500:]])
                feature = feature[:, :len(token[0]), :]
                result_token.extend(token[0][1:])
                result_feature[:, i * 500 + 1:, :] = feature[:, 1:, :]
    assert len(result_token) == len(input_list) + 2
    return result_feature[:, 1:-1, :]


class punctuate_model:
    """ 标点修正"""

    def __init__(self, path):
        # Create local graph and use it in the session
        self.path = path
        self.graph = tf.Graph()
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.sess = tf.Session(config=self.config, graph=self.graph)
        with self.graph.as_default():
            with gfile.FastGFile(os.path.join(self.path, 'model.pb'), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')  # 导入计算图
                # 需要有一个初始化的过程
                self.sess.run(tf.global_variables_initializer())
                # 输入
                self.word_embeddings = self.sess.graph.get_tensor_by_name("word_embeddings:0")
                self.sequence_lengths = self.sess.graph.get_tensor_by_name('sequence_lengths:0')
                # 输出
                self.pred_ids = self.sess.graph.get_tensor_by_name('output:0')
                logger.info(path + '  model.pb has been loaded!')
                self.max_length = int(self.pred_ids.shape[-1])
                self.label = self.transfer_label()

    def transfer_label(self):
        """
        获取标签转换字典
        :return: 标签转换字典
        """
        with open(os.path.join(self.path, 'label_2_ids.json'), 'r', encoding='utf-8')as f:
            label_2_ids = json.load(f)
        id_2_labels = {}
        for key in label_2_ids.keys():
            id_2_labels[label_2_ids[key]] = key
        return id_2_labels

    def get_result(self, feature, sentence, id):
        """
        获取单个句子的标点预测结果
        :param sentence:句子
        :param feature:句子的特征
        :param id:句子的id
        :return:修正结果及特征
        """
        feature_input = np.zeros([1, self.max_length, 312], dtype=np.float32)
        feature_input[:, :len(sentence), :] = feature
        dict_in = {self.word_embeddings: feature_input,
                   self.sequence_lengths: [len(sentence)]}
        pred_ids_out = self.sess.run(self.pred_ids, feed_dict=dict_in)
        new_tokens = []
        new_labels = []
        for t, p in zip(sentence, pred_ids_out[0]):
            new_tokens.append(t)
            new_labels.append(self.label[p])

        ner_results = {'id': id,
                       'result': new_labels,
                       'sentence': sentence,
                       'feature': feature}
        return get_punctuations(ner_results)

    def predict(self, datas, is_global=False):
        """
        对段落标点进行修正
        :param datas:段落列表
        :param is_global:是否为全局特征的标点修正，如果为全局特征的标点修正，需要返回段落的特征
        :return:当is_global为True时，返回标点修正结果和段落特征，否则只返回标点修正结果
        """
        result_list = []
        for data in datas:
            length = len(data['sentence']['context'])
            feature = get_features(data['sentence']['context'])
            if length <= self.max_length:
                result, feature_output = self.get_result(feature, data['sentence']['context'], data["id"])
                if is_global:
                    result['feature'] = feature_output
                result_list.append(result)
            else:
                num2 = int(length / self.max_length) + 1
                temp_result_list = []
                for i in range(num2):
                    if i != num2:
                        feature_input = feature[:, i * self.max_length:(i + 1) * self.max_length, :]
                        data_input = data['sentence']['context'][i * self.max_length:(i + 1) * self.max_length]
                        result, feature_output = self.get_result(feature_input, data_input, data["id"])
                        temp_result_list.append(result)
                        feature[:, i * self.max_length:(i + 1) * self.max_length, :] = feature_output
                    else:
                        feature_input = feature[:, i * self.max_length:, :]
                        data_input = data['sentence']['context'][i * self.max_length:]
                        result, feature_output = self.get_result(feature_input, data_input, data["id"])
                        temp_result_list.append(result)
                        feature[:, i * self.max_length:, :] = feature_output
                result = temp_result_list[0]
                for temp_result in temp_result_list[1:]:
                    result['result'] += temp_result['result']
                    result['context'] += temp_result['context']
                if is_global:
                    result['feature'] = feature
                result_list.append(result)
        return result_list
