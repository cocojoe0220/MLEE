import traceback
# import psutil
# import pynvml
from container import port
from flask import jsonify, Flask, request
from exception import OutError, check, return_error, return_success
from log import logger
from serving import serving_train, query_train_reuslt, serving_paragraph_global, Feature

app = Flask(__name__)

train_status_info = {}

# 预标准模型训练
@app.route('/semr_train', methods=['GET', 'POST'])
def semr_train():
    train_id = '-1'
    try:
        train_id, label_id, algo_type, task_id, cv_flag, _ = check.check_param_semr_train(request)
        serving_train(train_id,
                      label_id,
                      algo_type,
                      task_id,
                      cv_flag,
                      train_status_info)
        return return_success(train_id)
    except OutError as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        train_status_info[train_id] = 3
        return return_error(train_id,
                            e.error_message)
    except Exception as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        train_status_info[train_id] = 3
        return return_error(train_id,
                            str(e))


# # 查看CPU使用情况
# @app.route("/memory", methods=['GET', 'POST'])
# def cpu_memory():
#     try:
#         mem = psutil.virtual_memory()
#         memory_json = {"used_percent": mem.percent}
#         return jsonify(memory_json)
#     except Exception as e:
#         error_message = {
#             "error_message": str(e)
#         }
#         return error_message


# # 查看GPU使用情况
# @app.route("/gpuinfo", methods=['GET', 'POST'])
# def gpu_memory():
#     try:
#         pynvml.nvmlInit()
#         handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#         util = pynvml.nvmlDeviceGetUtilizationRates(handle)
#         gpu_util = util.gpu
#         real_mem_util = util.memory
#         meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         mem_used = round(meminfo.used * 100 / meminfo.total, 1)
#         memory_json = {
#             "gpu_util": gpu_util,
#             "real_mem_util": real_mem_util,
#             "mem_used": mem_used}
#         return jsonify(memory_json)
#     except Exception as e:
#         error_message = {
#             "error_message": str(e)
#         }
#         return jsonify(error_message)

# 检测训练服务心跳
@app.route("/health", methods=['GET', 'POST'])
def health():
    result = {
        "status": 1
    }
    # logger.info(results)
    return jsonify(result)

# 查询训练任务状态
@app.route('/train_status', methods=['GET', 'POST'])
def query_train_status():
    results = {
        "status": True,
        "result": train_status_info
    }
    # logger.info(results)
    return jsonify(results)

# 提取全局特征
@app.route('/paragraph_bert_feature', methods=['GET', 'POST'])
def get_paragraph_feature():
    try:
        paragraph_ids, classify_label_id = check.check_param_paragraph_feature(request)
        result = Feature.get_paragraph_feature_modify(paragraph_ids)
        return jsonify(result)
    except OutError as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        result = {
            "msg": str(e),
            "status": False
        }
        return jsonify(result)
    except Exception as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        result = {
            "msg": str(e),
            "status": False
        }
        return jsonify(result)

# 全局模型训练
@app.route('/paragraph_global_train', methods=['GET', 'POST'])
def paragraph_train_global():
    train_id = '-1'
    try:
        train_id, label_id, algo_type, task_id, _, global_version = check.check_param_semr_train(request)
        serving_paragraph_global(train_id,
                      label_id,
                      algo_type,
                      task_id,
                      train_status_info,
                      global_version)
        return return_success(train_id)
    except OutError as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        train_status_info[train_id] = 3
        return return_error(train_id,
                            e.error_message)
    except Exception as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        train_status_info[train_id] = 3
        return return_error(train_id,
                            str(e))

# 查看模型训练结果
@app.route('/train_result', methods=['GET', 'POST'])
def query_train_result():
    train_id = '-1'
    try:
        train_id, label_id, algo_type, task_id, global_version = check.check_param_train_result(request)
        result = query_train_reuslt(train_id,
                                    label_id,
                                    algo_type,
                                    task_id,
                                    global_version)
        return jsonify(result)
    except OutError as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        return return_error(train_id,
                            e.error_message)
    except Exception as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        return return_error(train_id,
                            str(e))


if __name__ == '__main__':
    app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
    app.run(host='0.0.0.0', debug=False, port=port)
