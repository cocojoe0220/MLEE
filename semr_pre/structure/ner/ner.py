import tensorflow as tf
from tensorflow.python.platform import gfile
from structure.bert import fast_bert
import json
import os
import numpy as np
import re
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


def get_features(data):
    """
    提取句子的特征，并将其中的特殊字符用“$”替换
    :param data:句子数据
    :return:句子的特征
    """
    data = re.sub(r'\r|\t|\n| ', '$', data)
    data = process_str(data)
    input_list = list(data)
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


class ner_model:
    """ 实体识别预测"""

    def __init__(self, path):
        # Create local graph and use it in the session
        self.path = path
        self.graph = tf.Graph()
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.1
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
                logger.info(path + ' model.pb has been loaded!')
                self.max_length = int(self.pred_ids.shape[-1])
                self.label = self.transfer_label()

    def transfer_label(self):
        """
        获取标签转换字典
        :return:标签转换字典
        """
        with open(os.path.join(self.path, 'label_2_ids.json'), 'r', encoding='utf-8')as f:
            label_2_ids = json.load(f)
        id_2_labels = {}
        for key in label_2_ids.keys():
            id_2_labels[label_2_ids[key]] = key
        return id_2_labels

    def predict_one(self, sentence, feature, id, ori_offset):
        """
        对单个句子进行实体识别预测
        :param sentence:输入的句子
        :param feature:输入的特征
        :param id:输入的句子id
        :param ori_offset:该句子第一个字的索引
        :return:句子的实体识别结果
        """
        feature_input = np.zeros([1, self.max_length, 312], dtype=np.float32)
        feature_input[:, :len(sentence), :] = feature
        dict_in = {self.word_embeddings: feature_input,
                   self.sequence_lengths: [len(sentence)]}
        pred_ids_out = self.sess.run(self.pred_ids, feed_dict=dict_in)

        entity_dic = {'content': ''}
        entity_list = []
        for i, word in enumerate(sentence):
            word_label = self.label[pred_ids_out[0][i]]
            if word_label != 'O':
                label_name = word_label[2:]
                if 'B-' in word_label:
                    if len(entity_dic['content']) != 0 and 'label' in entity_dic.keys():
                        entity_list.append(entity_dic)
                    entity_dic = {
                        'content': word,
                        'offset': i + ori_offset,
                        'label': label_name
                    }
                else:
                    if i != 0:
                        before_word_label = self.label[pred_ids_out[0][i - 1]]
                        before_label_name = before_word_label[2:]
                        if label_name == before_label_name:
                            entity_dic['content'] += word
                        else:
                            if len(entity_dic['content']) != 0 and 'label' in entity_dic.keys():
                                entity_list.append(entity_dic)
                                entity_dic = {'content': ''}
            else:
                if len(entity_dic['content']) != 0 and 'label' in entity_dic.keys():
                    entity_list.append(entity_dic)
                    entity_dic = {'content': ''}

            if i + 1 == len(sentence):
                if len(entity_dic['content']) != 0 and 'label' in entity_dic.keys():
                    entity_list.append(entity_dic)

        entity_list_out = []

        for entity in entity_list:
            if len(entity['content'].replace(' ', '')) != 0:
                entity_list_out.append(entity)

        ner_results = {
            'id': id,
            'result': entity_list_out,
            'context': sentence
        }

        return ner_results

    def predict(self, datas):
        """
        对多个句子进行实体识别预测
        :param datas:包含多个句子和句子id的列表
        :return:包含多个句子及实体识别结果的列表
        """
        result_list = []
        for data in datas:
            feature = get_features(data['sentence']['context'])
            length = len(data['sentence']['context'])
            if length <= self.max_length:
                result_list.append(self.predict_one(data['sentence']['context'], feature, data['id'], 0))
            else:
                temp_result_list = []
                id = data['id']
                new_sentences = data['sentence']['context'].split('，')
                input_sentence = ''
                ori_offset = 0
                for sentence_id, new_sentence in enumerate(new_sentences):
                    if len(input_sentence + new_sentence) < self.max_length:
                        if sentence_id + 1 == len(new_sentences):
                            input_feature = feature[:, ori_offset:ori_offset + len(input_sentence), :]
                            temp_result_list.append(self.predict_one(input_sentence, input_feature, id, ori_offset))
                        else:
                            input_sentence += new_sentence + '，'
                    else:
                        input_feature = feature[:, ori_offset:ori_offset + len(input_sentence), :]
                        temp_result_list.append(self.predict_one(input_sentence, input_feature, id, ori_offset))
                        ori_offset += len(input_sentence)
                        input_sentence = new_sentence + '，'

                result_out = temp_result_list[0]
                for temp_result in temp_result_list[1:]:
                    result_out['result'].extend(temp_result['result'])
                result_list.append(result_out)
        return result_list
