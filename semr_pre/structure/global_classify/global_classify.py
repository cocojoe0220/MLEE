import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import json
import numpy as np
from log import logger

class global_classify_model:
    """ 基于全局特征的句子类别预测 """

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
                logger.info(path + '  model.pb has been loaded!')
                self.max_length = int(self.word_embeddings.shape[1])
                self.id_2_label = self.transfer_label()

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

    def predict(self, datas):
        """
        预测句子类别，当句子过长时，会裁剪掉后面的内容
        :param datas:句子列表
        :return:句子类别列表
        """
        result_list = []
        feature_input = np.zeros([len(datas), self.max_length, 312])
        length_input = []
        for i, data in enumerate(datas):
            sentence_length = len(data['sentence']['context'])
            if sentence_length < self.max_length:
                length_input.append(sentence_length)
                feature_input[i, :sentence_length, :] = data['sentence']['feature']
            else:
                length_input.append(self.max_length)
                feature_input[i, :self.max_length, :] = data['sentence']['feature'][:self.max_length, :]

        dict_in = {self.word_embeddings: feature_input,
                   self.sequence_lengths: length_input}
        out = self.sess.run(self.pred_ids, feed_dict=dict_in)

        for i, data in enumerate(datas):
            result = {'id': data["id"],
                      'context': data['sentence']['context'],
                      'label': self.id_2_label[out[i]],
                      'feature': data['sentence']['feature']}
            result_list.append(result)
        return result_list
