import traceback
from flask import Flask, request
from exception import OutError, check, return_success, return_error
from log import logger
from container import port
from serving import semr_pre, paragraph_global_serve

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 预标注计算
@app.route('/semr_pre', methods=['GET', 'POST'])
def semr_structure_pre():
    try:
        train_id, label_id, algo_type, task_id, data_array = check.check_param_semr_pre(request)
        results = semr_pre(train_id, label_id, algo_type, task_id, data_array)
        return return_success(results)
    except OutError as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        return return_error(e.error_message)
    except Exception as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        return return_error(str(e))

# 全局计算
@app.route('/paragraph_global_serve', methods=['GET', 'POST'])
def semr_paragraph_global_serve():
    try:
        task_id, data_array, global_version = check.check_param_semr_serve(request)
        results = paragraph_global_serve(task_id, data_array, global_version)
        return return_success(results)
    except OutError as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        return return_error(e.error_message)
    except Exception as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        return return_error(str(e))


if __name__ == '__main__':
    app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
    app.run(host='0.0.0.0', debug=False, port=port)
