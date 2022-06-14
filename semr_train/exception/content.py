class Error:
    def __init__(self, error_message):
        self.msg = error_message
        self.status = False

class Success:
    def __init__(self, success_message):
        self.success = success_message

class OutError(Exception):
    def __init__(self, error_message):
        self.error_message = error_message

class TrainReturn:
    def __init__(self, train_id, errorMsg, result):
        self.train_id = int(train_id)
        self.msg = errorMsg
        self.status = bool(result)

class TrainMQ:
    def __init__(self, train_id, state, errorMsg, resultDataUrl):
        self.train_id = int(train_id)
        self.taskState = int(state)
        self.errorMsg = errorMsg
        self.resultDataUrl = resultDataUrl