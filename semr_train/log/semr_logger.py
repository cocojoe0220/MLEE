import os
import logging
import errno
import json
import logging.config

def init_logger(name):
    model_dir = os.path.join('./log/logger')
    make_sure_path_exists(model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    log = logging.getLogger(name)
    with open("./log/log_conf.json") as f:
        config = json.loads(f.read())
        logging.config.dictConfig(config)
    return log

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

logger = init_logger('semr_train')