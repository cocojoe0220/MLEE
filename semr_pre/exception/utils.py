from flask import jsonify
from .content import Error, Success
import json
from log import logger


def return_error(error_message):
    message = Error(error_message).__dict__
    logger.error(message)
    return jsonify(message)

def return_success(result):
    message = Success(result).__dict__
    logger.debug(message)
    return json.dumps(message, indent=4, ensure_ascii=False, sort_keys=False)