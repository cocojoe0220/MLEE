import os
import json
from container import model_address
from exception import OutError
from exception import ErrorMessage
from log import logger
from concurrent.futures import ThreadPoolExecutor
from structure import train_classify_servable, train_ner_servable, train_punctuate_servable, train_global_ner_servable, train_global_classify_servable
from .util import Reader

executor = ThreadPoolExecutor(3)


def serving_train(train_id, label_id, algo_type, task_id, cv_flag, train_status_info):
    # 测试服务器连接是否可用
    Reader.test_postgres_contect()
    train_status_info[train_id] = 1
    if algo_type == '0':
        train_corpus = Reader.read_punctuation_corpus(task_id)
        # train_punctuate_servable(train_id,
        #                          label_id,
        #                          algo_type,
        #                          task_id,
        #                          -1,
        #                          cv_flag,
        #                          train_corpus,
        #                          train_status_info)

        executor.submit(train_punctuate_servable,
                        train_id,
                        label_id,
                        algo_type,
                        task_id,
                        -1,
                        cv_flag,
                        train_corpus,
                        train_status_info)

    # 训练句子分类(classify)
    elif algo_type == '1':
        train_corpus = Reader.read_classify_corpus(task_id)
        logger.info('train data lengths :' + str(len(train_corpus)))
        train_classify_servable(train_id,
                                label_id,
                                algo_type,
                                task_id,
                                cv_flag,
                                train_corpus,
                                train_status_info)
        # executor.submit(train_classify_servable,
        #                 train_id,
        #                 label_id,
        #                 algo_type,
        #                 task_id,
        #                 cv_flag,
        #                 train_corpus,
        #                 train_status_info)
    # 训练实体识别模型
    elif algo_type == '2':
        # 读取标签的类型
        label_type = int(Reader.read_label_type(label_id))
        train_corpus = []
        # 训练句子分类的标签
        if label_type == 2:
            train_corpus = Reader.read_corpus_label_type_two(label_id)

        # 训练实体标签
        elif label_type in [3, 4]:
            train_corpus = Reader.read_corpus_label_type_three(label_id)

        # 属性
        elif label_type == 5:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0003_01)
        # 语料为空
        if len(train_corpus) == 0:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0003_02)

        # train_ner_servable(train_id,
        #                    label_id,
        #                    algo_type,
        #                    task_id,
        #                    cv_flag,
        #                    train_corpus,
        #                    train_status_info)
        # import json
        # with open('train_corpus.txt', 'w', encoding='utf-8') as f:
        #     f.write(json.dumps({
        #         "train_corpus": train_corpus
        #     }, ensure_ascii=False))
        executor.submit(train_ner_servable,
                        train_id,
                        label_id,
                        algo_type,
                        task_id,
                        cv_flag,
                        train_corpus,
                        train_status_info)
    else:
        raise OutError("Algo type is error!")

def serving_paragraph_global(train_id, label_id, algo_type, task_id, train_status_info, global_version):
    '''
    段落级别的全局特征计算
    :param train_id: 训练任务id
    :param label_id: 标签id
    :param algo_type: 算法类型
    :param task_id: 任务id
    :param train_status_info: 训练任务状态
    :param global_version: 全局版本号
    :return
    '''
    # 测试服务器连接是否可用
    Reader.test_postgres_contect()
    train_status_info[train_id] = 1
    if algo_type == '0':
        train_corpus = Reader.read_punctuation_corpus(task_id)
        logger.info('train data lengths :' + str(len(train_corpus)))
        # train_punctuate_servable(train_id,
        #                          label_id,
        #                          algo_type,
        #                          task_id,
        #                          global_version,
        #                          False,
        #                          train_corpus,
        #                          train_status_info)
        executor.submit(train_punctuate_servable,
                        train_id,
                        label_id,
                        algo_type,
                        task_id,
                        global_version,
                        False,
                        train_corpus,
                        train_status_info)

    # 训练句子分类(classify)
    elif algo_type == '1':
        train_corpus = Reader.search_paragraph_entity_corpus_train(label_id)
        logger.info('train data lengths :' + str(len(train_corpus['global_corpus'])))
        # train_global_classify_servable(train_id,
        #                         label_id,
        #                         algo_type,
        #                         task_id,
        #                         train_corpus,
        #                         train_status_info,
        #                         global_version)
        executor.submit(train_global_classify_servable,
                        train_id,
                        label_id,
                        algo_type,
                        task_id,
                        train_corpus,
                        train_status_info,
                        global_version)

    # 训练实体识别模型
    elif algo_type == '2':

        global_train_corpus = Reader.search_paragraph_entity_corpus_train(label_id)
        # a = global_train_corpus['global_corpus']
        logger.info('train data lengths :' + str(len(global_train_corpus['global_corpus'])))
        train_status_info[train_id] = 1
        # train_global_ner_servable(train_id,
        #                     label_id,
        #                     algo_type,
        #                     task_id,
        #                     global_train_corpus,
        #                     train_status_info,
        #                     global_version)
        executor.submit(train_global_ner_servable,
                        train_id,
                        label_id,
                        algo_type,
                        task_id,
                        global_train_corpus,
                        train_status_info,
                        global_version)
    else:
        raise OutError("Algo type is error!")

def query_train_reuslt(train_id, label_id, algo_type, task_id, global_version):
    if global_version is not None:
        train_result_path = os.path.join(model_address,
                                         task_id,
                                         'global',
                                         global_version,
                                         algo_type,
                                         label_id,
                                         train_id,
                                         'result.json')
    else:
        train_result_path = os.path.join(model_address,
                                         task_id,
                                         'pre',
                                         algo_type,
                                         label_id,
                                         train_id,
                                         'result.json')

    try:
        with open(train_result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        for k, v in result.items():
            if v != 0 and k in ['precision', 'recall', 'f1']:
                result[k] = round(v, 3)
        return {
            "train_id": train_id,
            "train_result": result
        }
    except FileNotFoundError:
        raise OutError(ErrorMessage.ERROR_MESSAGE_0003_03)
