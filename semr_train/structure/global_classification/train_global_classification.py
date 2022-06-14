import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.python.framework import graph_util
from .extract_global_features import save_features
import h5py
import os
import shutil
import time
from log import logger
from tensorflow.contrib.rnn import GRUCell


class classification_global_datas:
    """
    构建数据读取的类，在初始化时读取所有h5文件，再根据后续的后分的输入将数据划分成训练集和测试集
    """
    def __init__(self, path):
        self.x_list = []
        self.y_list = []
        self.data_length_list = []
        files = os.listdir(path)
        for file in files:
            if '.h5' in file:
                file_name = os.path.join(path, file)
                x, y, l = self.load_classification_h5(file_name)
                self.x_list.append(x)
                self.y_list.append(y)
                self.data_length_list.append(l)
        self.h5_num = len(self.x_list)

    def load_classification_h5(self, data_name):
        with h5py.File(data_name, 'r') as f:
            x = f['classification_global_datas']['features'][()]
            y = f['classification_global_datas']['labels'][()]
            l = f['classification_global_datas']['lengths'][()]
        return x, y, l

    def get_datas(self, val_i):
        """

        :param val_i: 测试集的编号，当其值过大时，则进行过拟合训练，测试集和训练集均为全集
        :return:分割后的数据集
        """
        train_x = []
        train_y = []
        train_length_list = []
        test_x = []
        test_y = []
        test_length_list = []
        if val_i <= self.h5_num:
            for i in range(self.h5_num):
                x = self.x_list[i]
                y = self.y_list[i]
                l = self.data_length_list[i]
                if i == val_i:
                    test_x = x
                    test_y = y
                    test_length_list = l
                else:
                    if len(train_x) == 0:
                        train_x = x
                        train_y = y
                        train_length_list = l
                    else:
                        train_x = np.append(train_x, x, axis=0)
                        train_y = np.append(train_y, y, axis=0)
                        train_length_list = np.append(train_length_list, l, axis=0)
        else:
            for i in range(self.h5_num):
                x = self.x_list[i]
                y = self.y_list[i]
                l = self.data_length_list[i]
                if len(train_x) == 0:
                    train_x = x
                    train_y = y
                    train_length_list = l
                else:
                    train_x = np.append(train_x, x, axis=0)
                    train_y = np.append(train_y, y, axis=0)
                    train_length_list = np.append(train_length_list, l, axis=0)
            test_x = train_x
            test_y = train_y
            test_length_list = train_length_list
        datas = {
            'train_x': train_x,
            'train_y': train_y,
            'test_length_list': test_length_list,
            'test_x': test_x,
            'test_y': test_y,
            'train_length_list': train_length_list
        }
        return datas


def train_single_global_classification(model_path, datas, label_nums, max_length=200, batch_size=10, epochs=30,
                                       learn_rate=0.001,
                                       early_stop_num_max=30):
    """

    :param model_path: 模型存储任务
    :param datas:训练数据
    :param label_nums:标签数量
    :param max_length:输入句子的最大长度
    :param batch_size:每次训练的数据量
    :param epochs:训练的迭代次数
    :param learn_rate:学习率
    :param early_stop_num_max:训练精度在early_stop_num_max个迭代没有上升之后，训练停止
    :return:最高精度
    """
    st = time.time()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    train_x = datas['train_x']
    train_y = datas['train_y']
    train_length_list = datas['train_length_list']
    test_x = datas['test_x']
    test_y = datas['test_y']
    test_length_list = datas['test_length_list']

    if np.shape(train_x)[0] < batch_size or np.shape(test_x)[0] < batch_size:
        batch_size = 1

    max_ac = 0
    ac_out = [0]
    early_stop_num = 0
    with tf.Session(config=config, graph=tf.Graph()) as sess:
        word_embeddings = tf.placeholder(tf.float32, shape=[None, max_length, 312], name="word_embeddings")
        labels = tf.placeholder(tf.int32, shape=[None, 1], name="labels")
        sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        base_lr = tf.constant(learn_rate)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.train.exponential_decay(base_lr, step_ph, epochs, 0.98, staircase=True)

        rnn_cell = GRUCell(312)
        outputs, logits = tf.nn.dynamic_rnn(
            cell=rnn_cell,  # 选择传入的cell
            inputs=word_embeddings,  # 传入的数据
            sequence_length=sequence_lengths,
            initial_state=None,  # 初始状态
            dtype=tf.float32,  # 数据类型
            time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)，这里根据image结构选择False
        )
        logits = tf.layers.dense(inputs=logits, units=label_nums, activation=None)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.one_hot(labels, depth=label_nums, dtype=tf.float32))
        pred = tf.arg_max(logits, 1, name='output')

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epochs):
            all_steps = int(np.shape(train_x)[0] / batch_size)
            for step in range(all_steps):
                train_feed_dict = {word_embeddings: train_x[step * batch_size:step * batch_size + batch_size, :, :],
                                   labels: train_y[step * batch_size:step * batch_size + batch_size, :],
                                   sequence_lengths: train_length_list[
                                                     step * batch_size:step * batch_size + batch_size],
                                   step_ph: epoch}
                loss_value, _ = sess.run([loss, train_op], feed_dict=train_feed_dict)
            step = all_steps - 1
            train_feed_dict = {word_embeddings: train_x[step * batch_size + batch_size:, :, :],
                               labels: train_y[step * batch_size + batch_size:, :],
                               sequence_lengths: train_length_list[step * batch_size + batch_size:],
                               step_ph: epoch}
            loss_value, _ = sess.run([loss, train_op], feed_dict=train_feed_dict)

            predict_label_list = []
            all_steps = int(np.shape(test_x)[0] / batch_size)
            for step in range(all_steps):
                test_feed_dict = {word_embeddings: test_x[step * batch_size:step * batch_size + batch_size, :, :],
                                  labels: test_y[step * batch_size:step * batch_size + batch_size, :],
                                  sequence_lengths: test_length_list[
                                                    step * batch_size:step * batch_size + batch_size],
                                  step_ph: epoch}
                pred_ids_out = sess.run(pred, feed_dict=test_feed_dict)
                predict_label_list.extend(pred_ids_out)
            step = all_steps - 1
            test_feed_dict = {word_embeddings: test_x[step * batch_size + batch_size:, :, :],
                              labels: test_y[step * batch_size + batch_size:, :],
                              sequence_lengths: test_length_list[step * batch_size + batch_size:],
                              step_ph: epoch}
            pred_ids_out = sess.run(pred, feed_dict=test_feed_dict)
            predict_label_list.extend(pred_ids_out)

            ac = accuracy_score(list(test_y), predict_label_list)
            ac = round(ac, 3)
            if ac > np.max(ac_out):
                logger.info('epoch: ' + str(epoch + 1) + '  max_accuracy: ' + str(ac))
                max_ac = ac
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
                with tf.gfile.FastGFile(model_path + 'model.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                early_stop_num = 0
            else:
                logger.info('epoch: ' + str(epoch + 1) + '  max_accuracy: ' + str(ac))
                early_stop_num += 1
            ac_out.append(ac)

            if epoch == 0:
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
                with tf.gfile.FastGFile(model_path + 'model.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
            if early_stop_num >= early_stop_num_max:
                epochs = epoch
                break
            if epoch >= early_stop_num_max and np.max(ac_out) == 0:
                epochs = epoch
                break

        result = {"algorithm": "classification",
                  "total_example": np.shape(train_x)[0] + np.shape(test_x)[0],
                  "train_example": np.shape(train_x)[0],
                  "dev_example": np.shape(test_x)[0],
                  "epoch_num": epochs,
                  "precision": max_ac,
                  "f1": max_ac,
                  "recall": max_ac,
                  "cost_time": int(time.time() - st)}

        with open(model_path + 'result.json', 'w', encoding='utf-8')as f:
            f.write(json.dumps(result, indent=4, ensure_ascii=False))

        return result


def train_global_classification(model_path, datas, max_length=300):
    data_path = os.path.join(model_path, 'data') + '/'
    model_path = model_path + '/'

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    label_nums = save_features(data_path, datas, max_length, kfold=5)
    shutil.copy(data_path + 'label_2_ids.json', model_path + 'label_2_ids.json')
    all_datas = classification_global_datas(data_path)

    one_datas = all_datas.get_datas(0)
    train_single_global_classification(model_path, one_datas, label_nums, max_length)
    shutil.rmtree(data_path)
