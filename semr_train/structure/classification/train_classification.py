import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.python.framework import graph_util
from .extract_features import save_features
import h5py
import os
import shutil
import time
from log import logger


class classification_datas:
    """
    构建数据读取的类，在初始化时读取所有h5文件，再根据后续的后分的输入将数据划分成训练集和测试集
    """
    def __init__(self, path):
        self.x_list = []
        self.y_list = []
        files = os.listdir(path)
        for file in files:
            if '.h5' in file:
                file_name = os.path.join(path, file)
                x, y = self.load_classification_h5(file_name)
                self.x_list.append(x)
                self.y_list.append(y)
        self.h5_num = len(self.x_list)

    def load_classification_h5(self, data_name):
        with h5py.File(data_name, 'r') as f:
            x = f['classification_data']['features'][()]
            y = f['classification_data']['labels'][()]
        return x, y

    def get_datas(self, val_i):
        """

        :param val_i: 测试集的编号，当其值过大时，则进行过拟合训练，测试集和训练集均为全集
        :return:
        """
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        if val_i <= self.h5_num:
            for i in range(self.h5_num):
                x = self.x_list[i]
                y = self.y_list[i]
                if i == val_i:
                    test_x = x
                    test_y = y
                else:
                    if len(train_x) == 0:
                        train_x = x
                        train_y = y
                    else:
                        train_x = np.append(train_x, x, axis=0)
                        train_y = np.append(train_y, y, axis=0)
        else:
            for i in range(self.h5_num):
                x = self.x_list[i]
                y = self.y_list[i]
                if len(train_x) == 0:
                    train_x = x
                    train_y = y
                else:
                    train_x = np.append(train_x, x, axis=0)
                    train_y = np.append(train_y, y, axis=0)
            test_x = train_x
            test_y = train_y
        datas = {
            'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y
        }
        return datas


def train_single_classification(model_path, datas, label_nums, batch_size=10, epochs=100, learn_rate=0.001,
                                early_stop_num_max=30):
    """

    :param model_path: 模型存储路径
    :param datas: 训练数据
    :param label_nums: 标签数量
    :param batch_size: 每个批次的数据量
    :param epochs: 每次训练的次数
    :param learn_rate: 学习率
    :param early_stop_num_max: 当 early_stop_num_max 个迭代精度不再上升，训练停止
    :return:最高精度
    """
    st = time.time()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    train_x = datas['train_x']
    train_y = datas['train_y']
    test_x = datas['test_x']
    test_y = datas['test_y']

    if np.shape(train_x)[0] < batch_size or np.shape(test_x)[0] < batch_size:
        batch_size = 1

    max_ac = 0
    ac_out = [0]
    early_stop_num = 0
    with tf.Session(config=config, graph=tf.Graph()) as sess:
        word_embeddings = tf.placeholder(tf.float32, shape=[None, 312], name="word_embeddings")
        labels = tf.placeholder(tf.int32, shape=[None, 1], name="labels")

        base_lr = tf.constant(learn_rate)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.train.exponential_decay(base_lr, step_ph, epochs, 0.98, staircase=True)

        logits = tf.layers.dense(inputs=word_embeddings, units=label_nums, activation=None)
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
                train_feed_dict = {word_embeddings: train_x[step * batch_size:step * batch_size + batch_size, :],
                                   labels: train_y[step * batch_size:step * batch_size + batch_size, :],
                                   step_ph: epoch}
                loss_value, _ = sess.run([loss, train_op], feed_dict=train_feed_dict)
            step = all_steps - 1
            train_feed_dict = {word_embeddings: train_x[step * batch_size + batch_size:, :],
                               labels: train_y[step * batch_size + batch_size:, :],
                               step_ph: epoch}
            loss_value, _ = sess.run([loss, train_op], feed_dict=train_feed_dict)

            predict_label_list = []
            all_steps = int(np.shape(test_x)[0] / batch_size)
            for step in range(all_steps):
                test_feed_dict = {word_embeddings: test_x[step * batch_size:step * batch_size + batch_size, :],
                                  labels: test_y[step * batch_size:step * batch_size + batch_size, :],
                                  step_ph: epoch}
                pred_ids_out = sess.run(pred, feed_dict=test_feed_dict)
                predict_label_list.extend(pred_ids_out)
            step = all_steps - 1
            test_feed_dict = {word_embeddings: test_x[step * batch_size + batch_size:, :],
                              labels: test_y[step * batch_size + batch_size:, :],
                              step_ph: epoch}
            pred_ids_out = sess.run(pred, feed_dict=test_feed_dict)
            predict_label_list.extend(pred_ids_out)

            ac = accuracy_score(list(test_y), predict_label_list)
            if ac != 0:
                ac = round(ac, 3)
            if ac > np.max(ac_out):
                logger.info('best_epoch: ' + str(epoch + 1) + ',  max_ac: ' + str(ac))
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


def train_classification(model_path, datas, cv=False):
    """

    :param model_path: 模型存储路径
    :param datas: 总体数据
    :param cv: 是否进行交叉验证，默认不进行交叉验证
    :return:
    """
    kfold = 5
    data_path = os.path.join(model_path, 'data') + '/'
    model_path = model_path + '/'
    st = time.time()

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    label_nums = save_features(data_path, datas, kfold)
    all_datas = classification_datas(data_path)
    shutil.copy(data_path + 'label_2_ids.json', model_path + 'label_2_ids.json')

    if cv:
        precision_list = []
        f1_list = []
        recall_list = []
        max_f1 = 0
        overall_result = {}

        for flod in range(kfold):
            one_datas = all_datas.get_datas(flod)
            model_path_i = model_path + 'fold_' + str(flod).zfill(2) + '/'
            if not os.path.exists(model_path_i):
                os.makedirs(model_path_i)
            result = train_single_classification(model_path_i, one_datas, label_nums)
            precision_list.append(result['precision'])
            f1_list.append(result['f1'])
            recall_list.append(result['recall'])
            if flod == 0:
                max_f1 = result['f1']
                shutil.copy(model_path_i + 'model.pb', model_path + 'model.pb')
                overall_result = result
            else:
                if max_f1 < result['f1']:
                    shutil.copy(model_path_i + 'model.pb', model_path + 'model.pb')
                    overall_result = result
            shutil.rmtree(model_path_i)
        shutil.rmtree(data_path)

        overall_result['cost_time'] = int(time.time() - st)
        overall_result['f1'] = np.mean(f1_list)
        overall_result['recall'] = np.mean(recall_list)
        overall_result['precision'] = np.mean(precision_list)
        with open(model_path + 'result.json', 'w', encoding='utf-8')as f:
            f.write(json.dumps(overall_result, indent=4, ensure_ascii=False))
    else:
        one_datas = all_datas.get_datas(0)
        train_single_classification(model_path, one_datas, label_nums)
        shutil.rmtree(data_path)
