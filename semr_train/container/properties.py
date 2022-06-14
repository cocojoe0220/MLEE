# 本地训练和服务模型绝对路径地址
model_address = '../deepcogni_tf_model'

# 训练端口号
port = 9002

postgre_dic = {
    "database": "cdp-ruxian",#cdp-huxi
    "user": "postgres",
    "password": "123456",
    "host": "10.101.15.254",
    "port": "5432",
}


bert_tiny_dict = {
    'vocab_file': 'structure/bert/model/vocab.txt',
    'init_checkpoint': 'structure/bert/model/bert_model.ckpt',
    'max_seq_length': 512,
    'layers': '-1',
    'bert_config_file': 'structure/bert/model/bert_config.json',
    'use_tpu': False,
    'use_one_hot_embeddings': False
}

feature_dict_path = 'structure/bert/model/token_features.json'

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"