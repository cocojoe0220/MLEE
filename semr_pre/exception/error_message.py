class ErrorMessage:
    # 获取参数异常
    pass
    # url发送请求异常
    ERROR_CODE_0001 = "Serving---0001"
    ERROR_MESSAGE_0001_01 = "Serving---Request String format error, cannot be load a json"
    ERROR_MESSAGE_0001_02 = "Serving---Request method isn't POST, please choose POST method"
    #
    ERROR_MESSAGE_0002_01 = "Serving---train_id empty"
    ERROR_MESSAGE_0002_02 = "Serving---train_id must be integer type"
    ERROR_MESSAGE_0002_03 = "Serving---train_id json format error"

    ERROR_MESSAGE_0002_04 = "Serving---algo_type empty"
    ERROR_MESSAGE_0002_05 = "Serving---algo_type must be integer type"
    ERROR_MESSAGE_0002_06 = "Serving---algo_type json format error"

    ERROR_MESSAGE_0002_07 = "Serving---label_id must be a integer type"
    ERROR_MESSAGE_0002_08 = "Serving---label_id json format error"

    ERROR_MESSAGE_0002_09 = "Serving---task_id empty"
    ERROR_MESSAGE_0002_10 = "Serving---task_id must be integer type"
    ERROR_MESSAGE_0002_11 = "Serving---task_id json format error"

    ERROR_MESSAGE_0002_12 = "Serving---data_array empty"
    ERROR_MESSAGE_0002_13 = "Serving---data_array must be list type"
    ERROR_MESSAGE_0002_14 = "Serving---data_array json format error"

    ERROR_MESSAGE_0002_15 = "Serving---data_array has empty element"