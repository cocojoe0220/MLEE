import tensorflow as tf
from tensorflow.python.platform import gfile
import json
import os
import numpy as np
from log import logger


class global_ner_model:
    """ 全局特征实体识别"""

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
                # self.dropout_pl = self.sess.graph.get_tensor_by_name('dropout:0')
                # 输出
                self.pred_ids = self.sess.graph.get_tensor_by_name('output:0')
                logger.info(path + '  model.pb has been loaded!')
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
        计算单个句子的实体识别结果
        :param sentence:句子
        :param feature:句子特征
        :param id:句子id
        :param ori_offset:句子第一个字在段落中的索引
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
            if len(entity['content'].replace(' ','')) != 0:
                entity_list_out.append(entity)

        ner_results = {
            'id': id,
            'result': entity_list_out,
            'context': sentence
        }

        return ner_results

    def predict(self, datas):
        """
        计算句子的实体识别结果
        :param datas:句子列表
        :return:实体识别结果列表
        """
        result_list = []
        for data in datas:
            feature = data['sentence']['feature']
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
                            input_feature = feature[ori_offset:ori_offset + len(input_sentence), :]
                            temp_result_list.append(self.predict_one(input_sentence, input_feature, id, ori_offset))
                        else:
                            input_sentence += new_sentence + '，'
                    else:
                        input_feature = feature[ori_offset:ori_offset + len(input_sentence), :]
                        temp_result_list.append(self.predict_one(input_sentence, input_feature, id, ori_offset))
                        ori_offset += len(input_sentence)
                        input_sentence = new_sentence + '，'

                result_out = temp_result_list[0]
                for temp_result in temp_result_list[1:]:
                    result_out['result'].extend(temp_result['result'])
                result_list.append(result_out)
        return result_list


