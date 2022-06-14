from structure import predict_servable
from .util import process_zero_type, process_one_type, process_two_type
from .process import  test_postgres_contect, predict_paragraph_global_label, load_global_model
import threading

lock = threading.Lock()
task_version = []

def semr_pre(train_id, label_id, algo_type, task_id, data_array):
    results = []
    if algo_type == '0':
        # 标点符号计算
        datas = process_zero_type(data_array)
        results = predict_servable(train_id,
                                   label_id,
                                   algo_type,
                                   task_id,
                                   datas)

    elif algo_type == '1':
        datas = process_one_type(data_array)
        results = predict_servable(train_id,
                                   label_id,
                                   algo_type,
                                   task_id,
                                   datas)

    elif algo_type == '2':
        datas = process_two_type(data_array)
        results = predict_servable(train_id,
                                   label_id,
                                   algo_type,
                                   task_id,
                                   datas)
    return results

def paragraph_global_serve(task_id, data_array, global_version):
    # 判断全局模型是否加载
    with lock:
        if task_id + '&' + str(global_version) not in task_version:
            task_version.append(task_id + '&' + str(global_version))
            load_global_model(task_id, global_version)
    test_postgres_contect()
    res = predict_paragraph_global_label(task_id, data_array, global_version)
    return res