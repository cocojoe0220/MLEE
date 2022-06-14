class ErrorMessage:
    # 获取参数异常
    pass
    # url发送请求异常
    ERROR_CODE_0001 = "Training---0001"
    ERROR_MESSAGE_0001_01 = "Training---request string format error, cannot be load a json"
    ERROR_MESSAGE_0001_02 = "Training---request method isn't post, please choose post method"
    #
    ERROR_CODE_0002 = "Training---0002"

    ERROR_MESSAGE_0002_01 = "Training---train_id empty"
    ERROR_MESSAGE_0002_02 = "Training---train_id must be integer type"
    ERROR_MESSAGE_0002_03 = "Training---train_id json format error"

    ERROR_MESSAGE_0002_04 = "Training---algo_type empty"
    ERROR_MESSAGE_0002_05 = "Training---algo_type must be integer type"
    ERROR_MESSAGE_0002_06 = "Training---algo_type json format error"

    ERROR_MESSAGE_0002_07 = "Training---label_id must be a integer type"
    ERROR_MESSAGE_0002_08 = "Training---label_id json format error"

    ERROR_MESSAGE_0002_09 = "Training---task_id empty"
    ERROR_MESSAGE_0002_10 = "Training---task_id must be integer type"
    ERROR_MESSAGE_0002_11 = "Training---task_id json format error"

    ERROR_MESSAGE_0002_12 = "Training---train_param must be a dict type"
    ERROR_MESSAGE_0002_13 = "Training---train_param json format error"

    ERROR_MESSAGE_0002_14 = "Training---sen_id_list empty"
    ERROR_MESSAGE_0002_15 = "Training---sen_id_list must be list type"
    ERROR_MESSAGE_0002_16 = "Training---sen_id_list json format error"

    ERROR_MESSAGE_0003_01 = "Training---label is attribute, can't be train"
    ERROR_MESSAGE_0003_02 = "Training---train corpus is empty"
    ERROR_MESSAGE_0003_03 = "Training---train result file not to be found"