def process_zero_type(data_array):
    datas = []
    for index, data in enumerate(data_array):
        datas.append({
            "id": data['id'],
            "sentence": {
                # "paragraph_id": 0,
                # "sentence_id": 0,
                # "context_offset": 0,
                "context": data['context']
            }
        })
    return datas


def process_one_type(data_array):
    datas = []
    for index, data in enumerate(data_array):
        datas.append({
            "id": data['id'],
            "sentence": {
                # "paragraph_id": 0,
                # "sentence_id": 0,
                # "context_offset": 0,
                "context": data['context']
            }
        })
    return datas


def process_two_type(data_array):
    datas = []
    for index, data in enumerate(data_array):
        datas.append({
            "id": data['id'],
            "sentence": {
                # "paragraph_id": 0,
                # "sentence_id": 0,
                # "context_offset": 0,
                "context": data['context']
            }
        })
    return datas


