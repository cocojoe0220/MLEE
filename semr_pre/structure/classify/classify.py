import tensorflow as tf
from tensorflow.python.platform import gfile
from structure.bert import fast_bert
import os
import json
from log import logger
import re


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


class classify_model:
    """ 基于局部特征的句子类别预测"""

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
                # 输出
                self.pred_ids = self.sess.graph.get_tensor_by_name('output:0')
                logger.info(path + '  model.pb has been loaded!')
                self.id_2_label = self.transfer_label()

    def transfer_label(self):
        """
        获取标签转换字典
        :return:标点转换字典
        """
        with open(os.path.join(self.path, 'label_2_ids.json'), 'r', encoding='utf-8')as f:
            label_2_ids = json.load(f)
        id_2_labels = {}
        for key in label_2_ids.keys():
            id_2_labels[label_2_ids[key]] = key
        return id_2_labels

    def predict(self, datas):
        """
        句子类别预测
        :param datas:句子列表
        :return:句子类别列表
        """
        results = []
        for data in datas:
            text = data['sentence']['context'].strip('\n').replace(' ', '')
            text = process_str(text)
            feature, _ = fast_bert.predict([list(text)])
            feature = feature[:, 0, :]
            dict_in = {self.word_embeddings: feature}
            out = self.sess.run(self.pred_ids, feed_dict=dict_in)
            result = {'id': data["id"],
                      'context': data['sentence']['context'],
                      'label': self.id_2_label[out[0]]}
            results.append(result)
        return results
