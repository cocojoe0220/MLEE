from .train_classification import train_classification
import os
from container import model_address
from log import logger
import traceback

# tensorflow日志打印级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_classify_servable(train_id, label_id, algo_type, task_id, cv_flag, train_corpus, train_status_info):
    try:
        model_path = os.path.join(model_address, task_id, 'pre', algo_type, label_id, train_id)
        train_classification(model_path, train_corpus, cv=cv_flag)
        logger.info("Train complete")
        train_status_info[train_id] = 2
    except Exception as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        logger.info("Train failed: " + str(e))
        train_status_info[train_id] = 3
