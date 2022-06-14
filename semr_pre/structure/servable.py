from .punctuate import punctuate_model
from .ner import ner_model
from .classify import classify_model
import os
from container import model_address
from .global_ner import global_ner_model
from .global_classify import global_classify_model

models = {}

# tensorflow日志打印级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def predict_servable(train_id, label_id, algo_type, task_id, predict_corpus):
    model_path = os.path.join(model_address, task_id, 'pre', algo_type, label_id, train_id)
    if model_path not in models.keys():
        if algo_type == '0':
            models[model_path] = punctuate_model(model_path)
        elif algo_type == '1':
            models[model_path] = classify_model(model_path)
        elif algo_type == '2':
            models[model_path] = ner_model(model_path)
    result = models[model_path].predict(predict_corpus)
    return result


def predict_global_servable(train_id, label_id, algo_type, task_id, predict_corpus, global_version):

    model_path = os.path.join(model_address, task_id, 'global', global_version, algo_type, label_id, train_id)
    if model_path not in models.keys():
        if algo_type == '0':
            models[model_path] = punctuate_model(model_path)
        elif algo_type == '1':
            models[model_path] = global_classify_model(model_path)
        elif algo_type == '2':
            models[model_path] = global_ner_model(model_path)
    if algo_type == '0':
        result = models[model_path].predict(predict_corpus, is_global=True)
    else:
        result = models[model_path].predict(predict_corpus)
    return result

def pre_load_global_model(train_id, label_id, algo_type, task_id, global_version):
    model_path = os.path.join(model_address, task_id, 'global', global_version, algo_type, label_id, train_id)
    if model_path not in models.keys():
        if algo_type == '0':
            models[model_path] = punctuate_model(model_path)
        elif algo_type == '1':
            models[model_path] = global_classify_model(model_path)
        elif algo_type == '2':
            models[model_path] = global_ner_model(model_path)
