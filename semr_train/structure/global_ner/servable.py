from .train_global_ner import train_global_ner
import os
from container import model_address
from log import logger
import traceback

# tensorflow日志打印级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_global_ner_servable(train_id, label_id, algo_type, task_id, train_corpus, train_status_info, global_version):
    try:
        model_path = os.path.join(model_address, task_id, 'global', global_version, algo_type, label_id, train_id)
        train_global_ner(model_path, train_corpus)
        logger.info("Train complete")
        train_status_info[train_id] = 4
    except Exception as e:
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
        logger.info("Train failed: " + str(e))
        train_status_info[train_id] = 3