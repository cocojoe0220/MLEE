from .content import TrainReturn
from flask import jsonify
from log import logger


def return_error(task_id, error_message):
    message = TrainReturn(task_id, error_message, False).__dict__
    logger.info("return error: " + str(message))
    return jsonify(message)

def return_success(task_id):
    message = TrainReturn(task_id, '', True).__dict__
    logger.info("return success: " + str(message))
    return jsonify(message)

