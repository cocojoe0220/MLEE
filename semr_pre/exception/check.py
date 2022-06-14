import json
from .content import OutError
from .error_message import ErrorMessage
from log import logger


class Check:
    def __init__(self):
        pass

    @staticmethod
    def check_param_semr_pre(request):
        try:
            requestData = request.get_data()
            json_data = json.loads(requestData)
            logger.info(json_data)
        except KeyError:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0001_01)
        if request.method != 'POST':
            raise OutError(ErrorMessage.ERROR_MESSAGE_0001_02)

        # 训练任务id校验
        try:
            train_id = json_data['train_id']
            if len(str(train_id)) == 0:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_01)
            if isinstance(train_id, int) is False:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_02)
        except KeyError:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0002_03)


        # 算法类型校验
        try:
            algo_type = json_data['algo_type']
            if len(str(algo_type)) == 0:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_04)
            if isinstance(algo_type, int) is False:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_05)
        except KeyError:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0002_06)

        # 标签id
        try:
            label_id = json_data['label_id']
            if len(str(label_id)) == 0:
                label_id = -1
            if isinstance(label_id, int) is False:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_07)
        except KeyError:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0002_08)

        # 标注任务id校验
        try:
            task_id = json_data['task_id']
            if len(str(task_id)) == 0:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_09)
            if isinstance(task_id, int) is False:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_10)
        except KeyError:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0002_11)

        # 任务参数
        try:
            data_array = json_data['data_array']
            if len(data_array) == 0:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_12)
            if isinstance(data_array, list) is False:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_13)
            for data in data_array:
                if len(data['context'].strip()) == 0:
                    raise OutError(ErrorMessage.ERROR_MESSAGE_0002_15)
        except KeyError:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0002_14)

        return str(train_id), str(label_id), str(algo_type), str(task_id), data_array

    @staticmethod
    def check_param_semr_serve(request):
        try:
            requestData = request.get_data()
            json_data = json.loads(requestData)
            logger.info(json_data)
        except:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0001_01)
        if request.method != 'POST':
            raise OutError(ErrorMessage.ERROR_MESSAGE_0001_02)

        # 标注任务id校验
        try:
            task_id = json_data['task_id']
            if len(str(task_id)) == 0:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_09)
            if isinstance(task_id, int) is False:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_10)
        except KeyError:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0002_11)

        # 标注任务id校验
        try:
            global_version = json_data['using_global_version'] if 'using_global_version' in json_data else None
            if isinstance(global_version, str) is False and global_version is not None:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_10)
        except KeyError:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0002_11)

        # 任务参数
        try:
            data_array = json_data['data_array']
            if len(data_array) == 0:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_12)
            if isinstance(data_array, list) is False:
                raise OutError(ErrorMessage.ERROR_MESSAGE_0002_13)
            for data in data_array:
                if len(data['context'].strip()) == 0:
                    raise OutError(ErrorMessage.ERROR_MESSAGE_0002_15)
        except KeyError:
            raise OutError(ErrorMessage.ERROR_MESSAGE_0002_14)

        return str(task_id), data_array , global_version
check = Check()