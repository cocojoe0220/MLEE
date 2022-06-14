import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib import crf
from tensorflow.python.framework import graph_util
from .extract_features import save_features
from .computer_accuracy import computer_f1
import h5py
import os
import shutil
import time
from log import logger


class punctuate_datas:
    """
    构建数据读取的类，在初始化时读取所有h5文件，再根据后续的后分的输入将数据划分成训练集和测试集
    """
    def __init__(self, path):
        self.x_list = []
        self.y_list = []
        self.real_labels_list = []
        self.data_length_list = []
        files = os.listdir(path)
        for file in files:
            if '.h5' in file:
                file_name = os.path.join(path, file)
                x, y, real_labels, length_list = self.load_ner_h5(file_name)
                self.x_list.append(x)
                self.y_list.append(y)
                self.real_labels_list.append(real_labels)
                self.data_length_list.append(length_list)
        self.h5_num = len(self.x_list)

    def load_ner_h5(self, data_name):
        with h5py.File(data_name, 'r') as f:
            x = f['punctuate_datas']['features'][()]
            y = f['punctuate_datas']['labels'][()]
            real_labels = f['punctuate_datas']['real_labels'][()]
            length_list = f['punctuate_datas']['lengths'][()]
        return x, y, real_labels, length_list

    def get_datas(self, val_i):
        """

        :param val_i: 测试集的编号，当其值过大时，则进行过拟合训练，测试集和训练集均为全集
        :return:切割后的训练集和测试集
        """
        train_x = []
        train_y = []
        train_length_list = []
        train_real_labels = []
        test_x = []
        test_y = []
        test_length_list = []
        test_real_labels = []
        if val_i <= self.h5_num:
            for i in range(self.h5_num):
                x = self.x_list[i]
                y = self.y_list[i]
                real_labels = self.real_labels_list[i]
                length_list = self.data_length_list[i]
                if i == val_i:
                    test_x = x
                    test_y = y
                    test_real_labels = real_labels
                    test_length_list = length_list
                else:
                    if len(train_x) == 0:
                        train_x = x
                        train_y = y
                        train_real_labels = real_labels
                        train_length_list = length_list
                    else:
                        train_x = np.append(train_x, x, axis=0)
                        train_y = np.append(train_y, y, axis=0)
                        train_real_labels = np.append(train_real_labels, real_labels, axis=0)
                        train_length_list = np.append(train_length_list, length_list, axis=0)
        else:
            for i in range(self.h5_num):
                x = self.x_list[i]
                y = self.y_list[i]
                real_labels = self.real_labels_list[i]
                length_list = self.data_length_list[i]
                if len(train_x) == 0:
                    train_x = x
                    train_y = y
                    train_real_labels = real_labels
                    train_length_list = length_list
                else:
                    train_x = np.append(train_x, x, axis=0)
                    train_y = np.append(train_y, y, axis=0)
                    train_real_labels = np.append(train_real_labels, real_labels, axis=0)
                    train_length_list = np.append(train_length_list, length_list, axis=0)
            test_x = train_x
            test_y = train_y
            test_real_labels = train_real_labels
            test_length_list = train_length_list

        datas = {
            'train_x': train_x,
            'train_y': train_y,
            'train_real_labels': train_real_labels,
            'train_length_list': train_length_list,
            'test_x': test_x,
            'test_y': test_y,
            'test_real_labels': test_real_labels,
            'test_length_list': test_length_list
        }
        return datas


def train_single_punctuate(model_path, datas, id_2_labels, batch_size=10, epochs=30, learn_rate=0.001, max_length=500,
                           early_stop_num_max=10):
    """

    :param model_path:模型存储路径
    :param datas:数据集
    :param id_2_labels:标签映射字典
    :param batch_size:每次迭代训练的数据量
    :param epochs:训练的迭代次数
    :param learn_rate:学习率
    :param max_length:句子最大长度
    :param early_stop_num_max:当early_stop_num_max个迭代，精度没有提高，训练停止
    :return:最高精度
    """
    st = time.time()
    test_x = datas['test_x']
    test_y = datas['test_y']
    test_length_list = datas['test_length_list']
    test_real_labels = datas['test_real_labels']
    train_x = datas['train_x']
    train_y = datas['train_y']
    train_length_list = datas['train_length_list']
    # train_real_labels = datas['train_real_labels']
    label_nums = len(id_2_labels.keys())
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session(config=config, graph=tf.Graph()) as sess:

        word_embeddings = tf.placeholder(tf.float32, shape=[None, None, 312], name="word_embeddings")
        labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        base_lr = tf.constant(learn_rate)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())

        learning_rate = tf.train.exponential_decay(base_lr, step_ph, epochs, 0.98, staircase=True)

        with tf.variable_scope("bi-gru"):

            cell_fw = GRUCell(312)
            cell_bw = GRUCell(312)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=word_embeddings,
                sequence_length=sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        logits = tf.layers.dense(inputs=output, units=label_nums, activation=None)
        log_likelihood, transition_params = crf_log_likelihood(inputs=logits,
                                                               tag_indices=labels,
                                                               sequence_lengths=sequence_lengths)
        loss = -tf.reduce_mean(log_likelihood)
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=transition_params,
                                     sequence_length=sequence_lengths)
        predict_save = tf.reshape(pred_ids, [-1, tf.constant(max_length)], name='output')
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)

        f1_out = [0]
        early_stop_num = 0
        if len(train_length_list) < batch_size or len(test_length_list) < batch_size:
            batch_size = 1
        max_out = {
            'total': {'precision': 0,
                      'recall': 0,
                      'f1': 0
                      }
        }
        for epoch in range(epochs):
            all_steps = int(len(train_length_list) / batch_size)
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

            label_list = []
            all_steps = int(len(test_length_list) / batch_size)
            test_length_index = 0
            for step in range(all_steps):
                test_feed_dict = {word_embeddings: test_x[step * batch_size:step * batch_size + batch_size, :, :],
                                  labels: test_y[step * batch_size:step * batch_size + batch_size, :],
                                  sequence_lengths: test_length_list[
                                                    step * batch_size:step * batch_size + batch_size],
                                  step_ph: epoch}

                pred_ids_out = sess.run(pred_ids, feed_dict=test_feed_dict)
                for kk in range(np.shape(pred_ids_out)[0]):
                    test_length_index = step * batch_size + kk
                    label_list.extend(pred_ids_out[kk, :test_length_list[test_length_index]])
            step = all_steps - 1
            test_feed_dict = {word_embeddings: test_x[step * batch_size + batch_size:, :, :],
                              labels: test_y[step * batch_size + batch_size:, :],
                              sequence_lengths: test_length_list[step * batch_size + batch_size:],
                              step_ph: epoch}
            pred_ids_out = sess.run(pred_ids, feed_dict=test_feed_dict)
            for kk2 in range(np.shape(pred_ids_out)[0]):
                label_list.extend(pred_ids_out[kk2, :test_length_list[test_length_index + 1 + kk2]])

            out, statistics = computer_f1(label_list, test_real_labels, id_2_labels)
            f1 = out['total']['f1']
            if epoch == 0:
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
                with tf.gfile.FastGFile(model_path + 'model.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

            if f1 > np.max(f1_out):
                logger.info('best_epoch: ' + str(epoch + 1) + ',  max_f1: ' + str(f1))
                max_out = out
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
                with tf.gfile.FastGFile(model_path + 'model.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                early_stop_num = 0
            else:
                logger.info('epoch: ' + str(epoch + 1) + ',  max_f1: ' + str(f1))
                early_stop_num += 1
            f1_out.append(f1)
            if early_stop_num >= early_stop_num_max:
                epochs = epoch
                break
            if epoch >= early_stop_num_max and np.max(f1_out) == 0.0:
                epochs = epoch
                break
        result = {"algorithm": "punctuate",
                  "total_example": np.shape(train_x)[0] + np.shape(test_x)[0],
                  "train_example": np.shape(train_x)[0],
                  "dev_example": np.shape(test_x)[0],
                  "epoch_num": epochs,
                  "precision": max_out['total']['precision'],
                  "recall": max_out['total']['recall'],
                  "f1": max_out['total']['f1'],
                  "cost_time": int(time.time() - st)}
        with open(model_path + 'result.json', 'w', encoding='utf-8')as f:
            f.write(json.dumps(result, indent=4, ensure_ascii=False))
    return result


def train_punctuate(model_path, datas, cv=False, max_length=300):
    """

    :param model_path:模型存储地址
    :param datas: 数据集
    :param cv: 是否进行交叉验证
    :param max_length: 句子最大长度
    :return:
    """
    st = time.time()
    kfold = 5
    data_path = os.path.join(model_path, 'data') + '/'
    model_path = model_path + '/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    save_features(data_path, datas, max_length, kfold)
    all_datas = punctuate_datas(data_path)
    with open(data_path + 'label_2_ids.json', 'r', encoding='utf-8')as f:
        label_2_ids = json.load(f)
    shutil.copy(data_path + 'label_2_ids.json', model_path + 'label_2_ids.json')
    id_2_labels = {}
    for key in label_2_ids.keys():
        id_2_labels[label_2_ids[key]] = key

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
            result = train_single_punctuate(model_path_i, one_datas, id_2_labels)
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
        train_single_punctuate(model_path, one_datas, id_2_labels)
        shutil.rmtree(data_path)
