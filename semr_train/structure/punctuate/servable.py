from .train_punctuate import train_punctuate
import os
from log import logger
from container import model_address
import traceback

# tensorflow日志打印级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_punctuate_servable(train_id, label_id, algo_type, task_id, global_version, cv_flag, train_corpus, train_status_info):
    try:
        if global_version != -1:
            model_path = os.path.join(model_address, task_id, 'global', global_version, algo_type, label_id, train_id)
        else:
            model_path = os.path.join(model_address, task_id, 'pre', algo_type, label_id, train_id)
        train_punctuate(model_path, train_corpus, cv=cv_flag)
        logger.info("Train complete")
        train_status_info[train_id] = 4 if global_version!= -1 else 2
    except Exception as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        logger.info("Train failed: " + str(e))
        train_status_info[train_id] = 3