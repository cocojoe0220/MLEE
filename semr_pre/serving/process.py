from exception import OutError
from structure import predict_global_servable, pre_load_global_model
from container import postgre, punctuate_type, classify_type, ner_type


def check_train_task_info(rows):
    if len(rows) != 1:
        raise OutError("Database train_task_info table error")


def process_input_data(data_array):
    datas = []
    for index, data in enumerate(data_array):
        datas.append({
            "id": index,
            "sentence": {
                "context": data['context']
            }
        })
    return datas


def process_global_input_data(data_array):
    datas = []
    for index, data in enumerate(data_array):
        datas.append({
            "id": index,
            "sentence": {
                "feature" : data['feature'],
                "context": data['context']
            }
        })
    return datas

class OutputElement:
    def __init__(self, index, label_id, content, entity_trees, sen_offset):
        self.id = index
        self.label_id = label_id
        self.content = content
        self.entityTrees = entity_trees
        self.offset = sen_offset


class OutputSubElement:
    def __init__(self, index, label_id, content, entity_trees, offset):
        self.id = index
        self.label_id = label_id
        self.content = content
        self.entityTrees = entity_trees
        self.offset = offset


def process_iteration_ner(index, label_id, data, global_feature, task_id, structure_res, offset, paragraph_offset, global_version):
    entity_trees = []
    ner_rows = postgre.search_label_id_info(label_id, task_id, global_version)
    if len(ner_rows) == 1:
        ner_datas = [{
            "id": 0,
            "sentence": {
                "context": data,
                "feature": global_feature[offset: len(data) + offset,:]
            }
        }]

        ner_results = predict_global_servable(str(ner_rows[0][0]),
                                       str(ner_rows[0][1]),
                                       str(ner_type),
                                       str(task_id),
                                       ner_datas,
                                       global_version)
        for ner_id, ner_res in enumerate(ner_results[0]['result']):
            process_iteration_ner(ner_id,
                                  ner_res['label'],
                                  ner_res['content'],
                                  global_feature,
                                  task_id,
                                  entity_trees,
                                  ner_res['offset'] + offset,
                                  paragraph_offset,
                                  global_version)
        structure_res.append(OutputSubElement(index,
                                              label_id,
                                              data,
                                              entity_trees,
                                              offset + paragraph_offset).__dict__)
    else:
        structure_res.append(OutputSubElement(index,
                                              label_id,
                                              data,
                                              [],
                                              offset + paragraph_offset).__dict__)

def test_postgres_contect():
    postgre.test_connect()


def predict_paragraph_global_label(task_id, data_array, global_version):

    punctuate_datas = process_input_data(data_array)
    punctuate_rows = postgre.search_algo_type_info(punctuate_type, task_id, global_version)
    check_train_task_info(punctuate_rows)
    punctuate_results = predict_global_servable(str(punctuate_rows[0][0]),
                                         str(punctuate_rows[0][1]),
                                         str(punctuate_type),
                                         str(task_id),
                                         punctuate_datas,
                                         global_version)
    structure_res = []
    paragraph_offset = 0
    for punctuate_res in punctuate_results:
        classify_sens = []
        sen_indexs = []
        offset = 0
        for index, char in enumerate(punctuate_res['result']):
            if char in ['。', '\n']:
                sen_indexs.append(index)
        if len(sen_indexs) == 0 or sen_indexs[-1] + 1 < len(punctuate_res['result']):
            sen_indexs.append(len(punctuate_res['result']))
        for sen_index in sen_indexs:
            classify_sens.append(punctuate_res['result'][offset:sen_index + 1])
            offset = sen_index + 1

        classify_data_array = []
        sen_offset = 0
        for i in range(len(classify_sens)):
            if len(classify_sens[i]):
                classify_data_array.append({
                    "context": classify_sens[i],
                    "feature": punctuate_res['feature'][0][sen_offset: sen_offset + len(classify_sens[i])]
                })
                sen_offset += len(classify_sens[i])

        classify_datas = process_global_input_data(classify_data_array)
        classify_rows = postgre.search_algo_type_info(classify_type, task_id, global_version)
        check_train_task_info(classify_rows)
        classify_results = predict_global_servable(str(classify_rows[0][0]),
                                                   str(classify_rows[0][1]),
                                                   str(classify_type),
                                                   str(task_id),
                                                   classify_datas,
                                                   global_version)

        paragraph_structure_res = []
        for index, classify_res in enumerate(classify_results):
            sen_structure_res = []
            sen_offset = 0
            process_iteration_ner(index,
                                  classify_res['label'],
                                  classify_res['context'],
                                  classify_res['feature'],
                                  task_id,
                                  sen_structure_res,
                                  sen_offset,
                                  paragraph_offset,
                                  global_version)

            paragraph_structure_res.append(OutputElement(index,
                                                         classify_res['label'],
                                                         classify_res['context'],
                                                         sen_structure_res[0]['entityTrees'],
                                                         sen_offset + paragraph_offset).__dict__)
            paragraph_offset += len(classify_res['context'])
        structure_res.append(paragraph_structure_res)
    return structure_res


def load_global_model(task_id, global_version):
    rows = postgre.search_global_version_model(task_id, global_version)
    for row in rows:
        pre_load_global_model(str(row[0]), str(row[1]), str(row[2]), task_id, global_version)