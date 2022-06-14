class Error:
    def __init__(self, error_message):
        self.status = False
        self.msg = error_message

class Success:
    def __init__(self, success_message):
        self.status = True
        self.success = success_message

class OutError(Exception):
    def __init__(self, error_message):
        self.error_message = error_message
