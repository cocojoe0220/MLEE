import re
import json
import numpy as np
from container import postgre
from log import logger


class Reader:
    def __init__(self):
        pass

    @staticmethod
    def read_punctuation_corpus(task_id):
        rows = postgre.search_punctuation_corpus(task_id)
        train_corpus = []
        for index, row in enumerate(rows):
            # corpus_context = re.sub('[\r\n\t]', '', row[0])
            corpus = {
                "id": index,
                "sentence": {
                    "paragraph_id": 0,
                    "sentence_id": 0,
                    "context_offset": 0,
                    "context": row[0]
                }
            }
            train_corpus.append(corpus)
        return train_corpus

    @staticmethod
    def read_classify_corpus(task_id):
        rows = postgre.search_classify_corpus(task_id)
        train_corpus = []
        for index, row in enumerate(rows):
            # corpus_context = re.sub('[\r\n\t]', '', row[2])
            corpus = {
                "id": index,
                "sentence": {
                    "paragraph_id": row[1],
                    "sentence_id": row[3],
                    "context": [row[2], str(row[0])]
                }
            }
            train_corpus.append(corpus)
        # with open('corpus.txt', 'w', encoding='utf-8') as f:
        #     f.write(json.dumps({"train_corpus":train_corpus}, indent=4, ensure_ascii=False))
        return train_corpus

    @staticmethod
    def read_label_type(label_id):
        rows = postgre.search_label_type(label_id)
        return rows[0][0]

    @staticmethod
    def read_corpus_label_type_two(label_id):
        train_corpus = []
        rows = postgre.search_data_from_sentence_info_two(label_id)
        sen_data = {}
        for row in rows:
            # corpus_context = re.sub('[\r\n\t]', '', row[1])
            corpus_context = row[1]

            sen_data[row[0]] = [corpus_context, row[2], row[3]]
        sen_index = 0
        for sen_id, sen_info in sen_data.items():
            sen_context = sen_info[0]
            train_context = [''] * len(sen_context)
            context = []
            paragraph_id = sen_info[1]
            sentence_id = sen_info[2]
            entity_rows = postgre.search_data_from_entity_info_two(sen_id)
            for row in entity_rows:
                label_id = row[0]
                entity_context = row[1]
                entity_offset = row[2]
                train_context[entity_offset] = [entity_context[0], 'B-' + str(label_id)]
                if len(entity_context) > 1:
                    for char_index, char in enumerate(entity_context[1:]):
                        train_context[entity_offset + char_index + 1] = [char, 'I-' + str(label_id)]
            for o_index, o_char in enumerate(train_context):
                if len(o_char) == 0:
                    context.append([sen_context[o_index], 'O'])
                else:
                    context.append(train_context[o_index])

            train_corpus.append({
                "id": sen_index,
                "sentence": {
                    "paragraph_id": paragraph_id,
                    "sentence_id": sentence_id,
                    "context_offset": 0,
                    "context": context
                }
            })
            sen_index += 1
        return train_corpus

    # !!!目前不针对于整句话的无限嵌套找索引=========#
    @staticmethod
    def read_corpus_label_type_three(label_id):
        train_corpus = []
        rows = postgre.search_entity_from_entity_info_three(label_id)
        entity_data = {}

        for row in rows:
            # corpus_context = re.sub('[\r\n\t]', '', row[1])
            corpus_context = row[1]
            entity_data[row[0]] = [corpus_context, row[2], row[3]]
        entity_index = 0
        for entity_id, entity_info in entity_data.items():
            context = []
            entity_context = entity_info[0]

            # 需要寻找句子位置 #
            # sen_id = entity_info[1]
            entity_offset = entity_info[2]
            train_context = [''] * len(entity_context)
            entity_rows = postgre.search_attribute_from_entity_info_three(entity_id)
            for row in entity_rows:
                label_id = row[0]
                attribute_context = row[1]
                entity_offset = row[2]
                train_context[entity_offset] = [attribute_context[0], 'B-' + str(label_id)]
                if len(attribute_context) > 1:
                    for char_index, char in enumerate(attribute_context[1:]):
                        train_context[entity_offset + char_index + 1] = [char, 'I-' + str(label_id)]

            for o_index, o_char in enumerate(train_context):
                if len(o_char) == 0:
                    context.append([entity_context[o_index], 'O'])
                else:
                    context.append(train_context[o_index])

            train_corpus.append({
                "id": entity_index,
                "sentence": {
                    "paragraph_id": 0,
                    "sentence_id": 0,
                    "context_offset": entity_offset,
                    "context": context
                }
            })
            entity_index += 1
        return train_corpus

    # @staticmethod
    # def search_sen_entity_corpus_train(label_id):
    #     '''
    #     从数据库读取全局特征,提供训练
    #     :param label_id:
    #     :return:
    #     '''
    #     global_rows = postgre.search_sen_global_corpus_from_label_id(label_id)
    #     global_corpus = []
    #     sen_ids = []
    #     for row in global_rows:
    #         global_corpus.append({
    #             "label_id": row[0],
    #             "label_content": row[1],
    #             "input_offset": row[2],
    #             "id": row[3]
    #         })
    #         if row[3] not in sen_ids:
    #             sen_ids.append(row[3])
    #     feature_rows = postgre.search_sentence_feature_from_sen_id(sen_ids)
    #     feature_corpus = []
    #     for row in feature_rows:
    #         feature_str = '{"feature" :' + row[2] + "}"
    #         feature_dic = json.loads(feature_str)
    #         feature_corpus.append({
    #             "id": row[0],
    #             "content": row[1],
    #             "feature": feature_dic['feature']
    #         })
    #     global_train_corpus = {
    #         "global_corpus": global_corpus,
    #         "feature_corpus": feature_corpus
    #     }
    #     return global_train_corpus

    @staticmethod
    def search_paragraph_entity_corpus_train(label_id):
        '''
        从数据库读取全局特征,提供训练
        :param label_id:
        :return:
        '''
        global_rows = postgre.search_entity_global_corpus_from_label_id(label_id)
        global_corpus = []
        paragraph_ids = []
        for row in global_rows:
            global_corpus.append({
                "label_id": row[0],
                "label_content": row[1],
                "input_offset": row[2],
                "id": row[3]
            })
            if row[3] not in paragraph_ids:
                paragraph_ids.append(row[3])
        feature_rows = postgre.search_paragraph_feature_from_paragraph_id(paragraph_ids)
        feature_corpus = []
        for row in feature_rows:
            feature_str = '{"feature" :' + row[2] + "}"
            feature_dic = json.loads(feature_str)
            feature_corpus.append({
                "id": row[0],
                "content": row[1],
                "feature": feature_dic['feature']
            })
        global_train_corpus = {
            "global_corpus": global_corpus,
            "feature_corpus": feature_corpus
        }
        return global_train_corpus

    # @staticmethod
    # def read_sentence_corpus_store(sen_ids):
    #     sen_rows = postgre.search_sentence_from_sen_ids(sen_ids)
    #     contexts = {}
    #     for sen_row in sen_rows:
    #         char_context = []
    #         sen_id = sen_row[0]
    #         sen_label_id = sen_row[1]
    #         sen_context = sen_row[2]
    #         for char in sen_context:
    #             char_context.append([char, str(sen_label_id)])
    #
    #         entity_rows = postgre.search_entity_from_sen_id(sen_id)
    #         for entity_row in entity_rows:
    #             Reader.read_entity_corpus(char_context, entity_row[0], entity_row[1], entity_row[2] + 0, entity_row[3])
    #
    #         contexts[sen_id] = char_context
    #     get_feature_corpus = {}
    #     for k, v in contexts.items():
    #         context_list = []
    #         for char in v:
    #             context_list.append('\t'.join(char))
    #         get_feature_corpus[k] = '\n'.join(context_list)
    #     return get_feature_corpus

    @staticmethod
    def read_paragraph_corpus_store(paragraph_ids):
        paragraph_rows = postgre.search_paragraph_from_paragraph_ids(paragraph_ids)
        logger.info('Gloabl train paragraph_ids length: ' + str(len(paragraph_rows)))
        contexts = {}
        for paragraph_row in paragraph_rows:
            paragraph_id = paragraph_row[0]
            single_context = []
            sen_rows = postgre.search_sen_from_paragraph_id(paragraph_id)
            for sen_row in sen_rows:
                char_context = []
                sen_id = sen_row[0]
                sen_label_id = sen_row[1]
                sen_context = sen_row[2] + '。'

                for char in sen_context:
                    # 如果句子是人工标注的
                    if sen_row[4] is not None and int(sen_row[4]) == 0:
                        char_context.append([char, str(sen_label_id)])
                    else:
                        char_context.append([char, "$$$" + str(sen_label_id)])
                # entity_rows = postgre.search_entity_from_sen_id(sen_id)
                # for entity_row in entity_rows:
                #     Reader.read_entity_corpus(char_context, entity_row[0], entity_row[1], entity_row[2] + 0,
                #                               entity_row[3])

                entity_rows = postgre.search_entity_from_sen_id(sen_id)
                for entity_row in entity_rows:
                    Reader.read_entity_corpus(char_context, entity_row[0], entity_row[1], entity_row[2] + 0,
                                              entity_row[3], entity_row[4], sen_row[5])

                single_context.extend(char_context)
            if len(single_context):
                contexts[paragraph_id] = single_context
        get_feature_corpus = {}
        # print(contexts)
        for k, v in contexts.items():
            context_list = []
            for char in v:
                context_list.append('\t'.join(char))
            get_feature_corpus[k] = '\n'.join(context_list)
        # print(get_feature_corpus)
        return get_feature_corpus

    @staticmethod
    def read_entity_corpus(char_context, entity_id, entity_context,
                           entity_offset, entity_label, is_sub_manual, is_manual):
        '''
        递归读取实体信息
        :param is_manual:
        :param is_sub_manual:
        :param char_context:
        :param entity_id:
        :param entity_context:
        :param entity_offset: 实体在句子的位置
        :param entity_label:
        :return:
        '''
        # index 字符在实体位置的offset
        if is_manual is not None and int(is_manual) == 0:
            for index, char in enumerate(entity_context):
                char_context[index + entity_offset].append(str(entity_label))
        else:
            for index, char in enumerate(entity_context):
                char_context[index + entity_offset].append("$$$" + str(entity_label))

        entity_rows = postgre.search_entity_from_entity_id(entity_id)
        if len(entity_rows):
            for entity_row in entity_rows:
                Reader.read_entity_corpus(char_context, entity_row[0], entity_row[1], entity_row[2] + entity_offset,
                                          entity_row[3], entity_row[4], is_sub_manual)

    @staticmethod
    def test_postgres_contect():
        '''
        检测数据库是否连接正常，无连接重新连接
        :return:
        '''
        postgre.test_connect()

    @staticmethod
    def get_paragraph_feature(paragraph_id):
        '''
        获取已经提取完成的段落特征
        :return:
        '''
        feature = None
        paragraph_feature_rows = postgre.search_paragraph_feature_from_paragraph_id([str(paragraph_id)])
        if len(paragraph_feature_rows) > 0:
            feature = paragraph_feature_rows[0][2]
        return feature

    @staticmethod
    def read_paragraph_corpus(paragraph_ids):
        logger.info('Gloabl train paragraph_ids length: ' + str(len(paragraph_ids)))

        paragraph_corpus = {}
        for paragraph_id in paragraph_ids:
            sen_rows = postgre.search_sen_by_paragraph_id(paragraph_id)
            paragraph_content = ""
            last_offset = -1
            for sen_row in sen_rows:
                if last_offset == int(sen_row[3]):
                    continue
                paragraph_content += sen_row[2] + '。'
                last_offset = int(sen_row[3])
            paragraph_content = re.sub(r'\r|\t|\n| ', '$', paragraph_content)
            paragraph_corpus[paragraph_id] = paragraph_content

        return paragraph_corpus
