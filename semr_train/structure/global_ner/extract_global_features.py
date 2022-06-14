import numpy as np
import h5py
import json
import os


def transfer_datas(path, name, labels_2_ids, feature_dic, global_corpus, max_length):
    lengths = []
    num = len(global_corpus)
    labels_out = np.zeros((num, max_length))
    feature_out = np.zeros((num, max_length, 312))
    real_labels = []
    for i, entity_sentence in enumerate(global_corpus):
        sentence_content = ''
        sen_id = entity_sentence['id']
        input_offset = entity_sentence['input_offset']
        label_content = json.loads(entity_sentence['label_content'])
        sentence_length = len(label_content.keys())
        if sentence_length > max_length:
            sentence_length = max_length
        lengths.append(sentence_length)
        for key in range(sentence_length):
            word, label = label_content[str(key)]
            sentence_content = sentence_content + word
            if label not in labels_2_ids.keys():
                labels_2_ids[label] = len(labels_2_ids.keys())
            labels_out[i, key] = labels_2_ids[label]
            real_labels.append(labels_2_ids[label])
        # print(feature_dic[sen_id]['content'])
        # print(len(feature_dic[sen_id]['content']))
        # print(np.shape(feature_dic[sen_id]['feature']))
        # print(np.shape(feature_dic[sen_id]['feature'][:,
        #                                       input_offset:input_offset + sentence_length, :]))
        # print(np.shape(feature_out[i, :sentence_length, :]))
        # print(sentence_content,feature_dic[sen_id]['content'][input_offset:input_offset + sentence_length])
        assert len(sentence_content) == len(feature_dic[sen_id]['content'][input_offset:input_offset + sentence_length])
        feature_out[i, :sentence_length, :] = feature_dic[sen_id]['feature'][:,
                                              input_offset:input_offset + sentence_length, :]

        if os.path.exists(path + name + ".h5"):
            os.remove(path + name + ".h5")
        with h5py.File(path + name + ".h5", "w") as f:
            out_datas = f.create_group('ner_datas')
            out_datas['features'] = feature_out
            out_datas['labels'] = labels_out
            out_datas['real_labels'] = real_labels
            out_datas['lengths'] = lengths
    return labels_2_ids


def save_features(data_path, datas, max_length, kfold=5):
    global_corpus = datas['global_corpus']
    feature_corpus = datas['feature_corpus']
    if len(global_corpus) == 1:
        global_corpus.append(global_corpus[0])
    np.random.shuffle(global_corpus)
    feature_dic = {}
    for feature in feature_corpus:
        feature_dic[feature['id']] = {
            'content': feature['content'],
            'feature': np.array(feature['feature'])
        }
    labels_2_ids = {'O': 0}

    start = 0
    for i in range(kfold):
        end = int(len(global_corpus) / kfold * (i + 1))
        one_fold_global_corpus = global_corpus[start:end]
        start = end
        name = 'fold_' + str(i).zfill(2)
        labels_2_ids = transfer_datas(data_path, name, labels_2_ids, feature_dic, one_fold_global_corpus, max_length)

    with open(data_path + 'label_2_ids.json', 'w', encoding='utf-8')as f:
        f.write(json.dumps(labels_2_ids, indent=4, ensure_ascii=False))
