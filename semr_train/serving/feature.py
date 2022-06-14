import re
import numpy as np
from structure import fast_bert_train, features_dict
from .util import Reader


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


class Feature:
    def __init__(self):
        pass

    @staticmethod
    def transfer_sen_label_2_entity(sentence_id, input_sentence):
        datas = []
        sentence_list = input_sentence.split('\n')

        length_list = []

        for sentence in sentence_list:
            sentence = sentence.strip('\n').split('\t')
            sentence[0] = re.sub(r'\r|\t|\n| ', '$', sentence[0])
            if sentence[0] != '':
                length_list.append(len(sentence))
                datas.append(sentence)

        j_nums = np.max(length_list)
        if j_nums == 2:
            j_nums = 3

        for data in datas:
            length = len(data)
            more_length = j_nums - length
            for i in range(more_length):
                data.append('O')

        input_label_id = {}

        results = []
        offset_list = []
        offset_out = 0
        label_id_list = []
        label_id_out = 0
        is_sub = True

        for j in range(1, j_nums - 1):
            labels = []
            sub_labels = []
            for offset, data in enumerate(datas):
                label_j = data[j]
                word = data[0]
                label_j_1 = data[j + 1]
                if label_j_1 != 'O':
                    if sub_labels == [] or label_j_1 != sub_labels[-1]:
                        label_j_sub = 'B-' + label_j_1
                    else:
                        label_j_sub = 'I-' + label_j_1
                    sub_labels.append(label_j_1)
                else:
                    label_j_sub = 'O'
                    sub_labels.append(label_j_sub)
                if labels == [] or label_j != labels[-1]:
                    if input_label_id != {} and label_id_out != 'O':
                        results.append(input_label_id)
                        offset_list.append(offset_out)
                        label_id_list.append(label_id_out)
                    num = 0
                    is_sub = True
                    input_label_id = {num: [word, label_j_sub]}
                    offset_out = offset
                    label_id_out = label_j
                else:
                    input_label_id[num] = [word, label_j_sub]
                if label_j_sub != 'O':
                    is_sub = True
                num += 1
                labels.append(data[j])

        if len(results) != 0:
            if input_label_id != results[-1] and label_id_out != 'O':
                results.append(input_label_id)
                offset_list.append(offset_out)
                label_id_list.append(label_id_out)
        else:
            if label_id_out != 'O':
                results.append(input_label_id)
                offset_list.append(offset_out)
                label_id_list.append(label_id_out)

        label1_list = []
        label2_list = []
        sentence_dic = {
            'sentence_content': '',
            'sen_id': sentence_id
        }
        if len(results) != 0:
            for offset, data in enumerate(datas):
                sentence_dic['sentence_content'] += data[0]

            sentence_feature = Feature.get_features(sentence_dic['sentence_content'])

            sentence_dic['sentence_feature'] = sentence_feature.tolist()
            label1_list.append(sentence_dic)
            for i, result in enumerate(results):
                entity_dic = {
                    'sentence_id': sentence_id,
                    'input_label_id': result,
                    'label_id': int(label_id_list[i]),
                    'input_offset': offset_list[i],
                }

                label2_list.append(entity_dic)

        return label1_list, label2_list

    @staticmethod
    def transfer_paragraph_label_2_entity(paragraph_label, sentence_id, input_paragraph):
        """
        提取全局特征，并进行实体识别、句子分类等数据的提取
        :param paragraph_label:句子分类任务的label_id
        :param sentence_id:段落的id
        :param input_sentence:输入的段落及标签
        :return:段落全局特征和实体识别、句子分类等任务的数据
        """
        datas = []
        paragraph_list = input_paragraph.split('\n')
        length_list = []

        for paragraph in paragraph_list:
            paragraph = paragraph.strip('\n').split('\t')
            paragraph[0] = re.sub(r'\r|\t|\n| ', '$', paragraph[0])
            if paragraph[0] != '':
                length_list.append(len(paragraph))
                datas.append(paragraph)

        j_nums = np.max(length_list)
        if j_nums == 2:
            j_nums = 3

        for data in datas:
            length = len(data)
            more_length = j_nums - length
            for i in range(more_length):
                data.append('O')
        new_datas = datas.copy()
        sentence_dic = {}
        sentence_list = []
        start_offset = 0
        for index in range(len(datas)):
            data = datas[index]
            sentence_list.append(data)
            if index + 1 != len(datas):
                if data[0] == '。':
                    sentence_dic[start_offset] = sentence_list
                    start_offset = index + 1
                    sentence_list = []
            else:
                sentence_dic[start_offset] = sentence_list

        results = []
        offset_list = []
        label_id_list = []
        for start_offset in sentence_dic.keys():
            datas = sentence_dic[start_offset]
            input_label_id = {}
            offset_out = start_offset
            label_id_out = 0
            is_sub = True

            for j in range(1, j_nums - 1):
                labels = []
                sub_labels = []
                for offset, data in enumerate(datas):
                    label_j = data[j]
                    word = data[0]
                    label_j_1 = data[j + 1]
                    if label_j_1 != 'O':
                        if sub_labels == [] or label_j_1 != sub_labels[-1]:
                            label_j_sub = 'B-' + label_j_1
                        else:
                            label_j_sub = 'I-' + label_j_1
                        sub_labels.append(label_j_1)
                    else:
                        label_j_sub = 'O'
                        sub_labels.append(label_j_sub)
                    if labels == [] or label_j != labels[-1]:
                        if input_label_id != {} and label_id_out != 'O':
                            results.append(input_label_id)
                            offset_list.append(offset_out)
                            label_id_list.append(label_id_out)
                        num = 0
                        is_sub = True
                        input_label_id = {num: [word, label_j_sub]}
                        offset_out = offset + start_offset
                        label_id_out = label_j
                    else:
                        input_label_id[num] = [word, label_j_sub]
                    if label_j_sub != 'O':
                        is_sub = True
                    num += 1
                    labels.append(data[j])

            if len(results) != 0:
                if input_label_id != results[-1] and label_id_out != 'O':
                    results.append(input_label_id)
                    offset_list.append(offset_out)
                    label_id_list.append(label_id_out)
            else:
                if label_id_out != 'O':
                    results.append(input_label_id)
                    offset_list.append(offset_out)
                    label_id_list.append(label_id_out)

        datas = new_datas
        label1_list = []
        label2_list = []
        sentence_dic = {
            'paragraph_content': '',
            'paragraph_id': sentence_id
        }
        if len(results) != 0:
            for offset, data in enumerate(datas):
                sentence_dic['paragraph_content'] += data[0]

            # 判断是否已经提取过该段落的特征
            sentence_feature = Feature.get_features(sentence_dic['paragraph_content'])
            sentence_dic['paragraph_feature'] = sentence_feature.tolist()

            label1_list.append(sentence_dic)
            for i, result in enumerate(results):
                entity_dic = {
                    'paragraph_id': sentence_id,
                    'input_label_id': result,
                    'label_id': int(str(label_id_list[i]).replace("$$$", "")),
                    'input_offset': offset_list[i],
                }
                label2_list.append(entity_dic)
        ###获取句子分类数据###

        sentence_content = datas[0][0]
        sentence_label = datas[0][1]
        sentence_offset = 0
        for offset in range(1, len(datas)):
            data = datas[offset]
            if offset + 1 != len(datas):
                after_data = datas[offset + 1]
                if data[1] == datas[offset - 1][1]:
                    sentence_content += data[0]
                    if data[0] == '。' and len(sentence_content) != 0:
                        classification_data = {
                            'paragraph_id': sentence_id,
                            'input_label_id': {'content': sentence_content, 'label': sentence_label},
                            'label_id': paragraph_label,
                            'input_offset': sentence_offset,
                        }
                        if "$$$" not in sentence_label:
                            label2_list.append(classification_data)
                        sentence_content = ''
                        sentence_label = after_data[1]
                        sentence_offset = offset + 1
                else:
                    if datas[offset - 1][0] != '。' and len(sentence_content) != 0:
                        classification_data = {
                            'paragraph_id': sentence_id,
                            'input_label_id': {'content': sentence_content, 'label': sentence_label},
                            'label_id': paragraph_label,
                            'input_offset': sentence_offset,
                        }
                        if "$$$" not in sentence_label:
                            label2_list.append(classification_data)
                    sentence_content = data[0]
                    sentence_label = data[1]
                    sentence_offset = offset
            else:
                sentence_content += data[0]
                classification_data = {
                    'paragraph_id': sentence_id,
                    'input_label_id': {'content': sentence_content, 'label': sentence_label},
                    'label_id': paragraph_label,
                    'input_offset': sentence_offset,
                }
                if "$$$" not in sentence_label:
                    label2_list.append(classification_data)

        return label1_list, label2_list

    @staticmethod
    def get_features(data):
        """
        提取段落特征，并在提取之前将数据中的标点进行替换，再在特征中使用字典中的标点特征进行替换
        :param data: 待提取特征的文本
        :return:文本的特征，其中没有cls和sep的特征，特征长度与文本长度一致
        """
        input_list = []
        index_dic = {}
        data = process_str(data)
        for i, word in enumerate(list(data)):
            if word in [',', '，', '。', '；']:
                input_list.append('@')
                index_dic[i + 1] = word
            else:
                input_list.append(word)
        if len(data) <= 500:
            feature, token = fast_bert_train.predict([input_list])
            result_token = token[0]
            result_feature = feature[:, :len(token[0]), :]
        else:
            result_feature = np.zeros([1, len(data) + 2, 312], dtype=np.float32)
            result_token = []
            num2 = int(len(input_list) / 500)
            for i in range(num2 + 1):
                if i != num2:
                    feature, token = fast_bert_train.predict([input_list[i * 500:(i + 1) * 500]])
                    feature = feature[:, :len(token[0]), :]
                    if i == 0:
                        result_token.extend(token[0][:-1])
                        result_feature[:, i * 500:(i + 1) * 500 + 1, :] = feature[:, :-1, :]
                    else:
                        result_token.extend(token[0][1:-1])
                        result_feature[:, i * 500 + 1:(i + 1) * 500 + 1, :] = feature[:, 1:-1, :]
                else:
                    feature, token = fast_bert_train.predict([input_list[i * 500:]])
                    feature = feature[:, :len(token[0]), :]
                    result_token.extend(token[0][1:])
                    result_feature[:, i * 500 + 1:, :] = feature[:, 1:, :]

        assert len(result_token) == len(input_list) + 2
        for index in index_dic.keys():
            assert '@' == result_token[index]
            result_feature[:, index, :] = features_dict[index_dic[index]]['feature']
        return result_feature[:, 1:-1, :]

    # @staticmethod
    # def get_sen_feature(sen_ids):
    #     '''
    #     提取句子全局特征
    #     :param sen_ids:
    #     :return:
    #     '''
    #     sen_corpus = Reader.read_sentence_corpus_store(sen_ids)
    #     sentence_feature = []
    #     global_corpus = []
    #
    #     for sentence_id in sen_corpus.keys():
    #         input_sentence = sen_corpus[sentence_id]
    #         label1_list, label2_list = Feature.transfer_sen_label_2_entity(sentence_id, input_sentence)
    #         sentence_feature.extend(label1_list)
    #         global_corpus.extend(label2_list)
    #     res = {
    #         "sentence_feature": sentence_feature,
    #         "global_corpus": global_corpus
    #     }
    #     return res

    @staticmethod
    def get_paragraph_feature(paragraph_ids, classify_label_id):
        '''
        提取段落全局特征
        :param paragraph_ids:
        :param classify_label_id:
        :return:
        '''
        Reader.test_postgres_contect()
        paragraph_corpus = Reader.read_paragraph_corpus_store(paragraph_ids)
        paragraph_feature = []
        global_corpus = []
        for paragraph_id in paragraph_corpus.keys():
            input_paragraph = paragraph_corpus[paragraph_id]
            label1_list, label2_list = Feature.transfer_paragraph_label_2_entity(classify_label_id, paragraph_id,
                                                                                 input_paragraph)
            paragraph_feature.extend(label1_list)
            global_corpus.extend(label2_list)
        res = {
            "paragraph_feature": paragraph_feature,
            "global_corpus": global_corpus
        }
        return res

    @staticmethod
    def get_paragraph_feature_modify(paragraph_ids):
        '''
        提取段落全局特征修改：只是提取bert特征没有其他操作
        :param paragraph_ids: 段落ID
        :return:
        '''
        Reader.test_postgres_contect()
        paragraph_corpus = Reader.read_paragraph_corpus(paragraph_ids)
        paragraph_feature = []

        for paragraph_id in paragraph_corpus.keys():
            sentence_dic = {
                'paragraph_content': paragraph_corpus[paragraph_id],
                'paragraph_id': paragraph_id
            }

            sentence_feature = Feature.get_features(sentence_dic['paragraph_content'])
            sentence_dic['paragraph_feature'] = sentence_feature.tolist()

            paragraph_feature.append(sentence_dic)
        res = {
            "paragraph_feature": paragraph_feature
        }
        return res
