def computer_f1(preds, labels, id_2_labels):
    assert len(preds) == len(labels)
    tps = {'total': 0}
    preds_true = {'total': 0}
    labels_true = {'total': 0}
    for i in range(len(preds)):
        pred = id_2_labels[preds[i]]
        label = id_2_labels[labels[i]]
        if pred[0] == 'B':
            preds_true['total'] += 1
            type_pred = pred.split('-')[-1]
            if type_pred not in preds_true.keys():
                preds_true[type_pred] = 1
            else:
                preds_true[type_pred] += 1
        if label[0] == 'B':
            labels_true['total'] += 1
            type_label = label.split('-')[-1]
            if type_label not in labels_true.keys():
                labels_true[type_label] = 1
            else:
                labels_true[type_label] += 1
            if label == pred:
                for j in range(1, len(labels) - i):
                    next_label = id_2_labels[labels[i + j]]
                    next_pred = id_2_labels[preds[i + j]]
                    if next_label == 'I-' + type_label:
                        if next_label != next_pred:
                            break
                    else:
                        tps['total'] += 1
                        if type_label not in tps.keys():
                            tps[type_label] = 1
                        else:
                            tps[type_label] += 1
                        break

    out = {}
    statistics = []
    for key in labels_true:
        label_true = labels_true[key]
        if key not in tps.keys():
            tp = 0
        else:
            tp = tps[key]
        if key not in preds_true.keys():
            pred_true = 0
        else:
            pred_true = preds_true[key]
        if tp != 0:
            precision = tp / pred_true
            recall = tp / label_true
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        precision = round(precision, 3)
        recall = round(recall, 3)
        f1 = round(f1, 3)
        out[key] = {'precision': precision,
                    'recall': recall,
                    'f1': f1}

        if key != 'total':
            statistics.append("{key}: Precision：{precision:.4f}; Recall：{recall:.4f}; F1-score：{f1:.4f}".format(key=key,
                                                                                                                precision=precision,
                                                                                                                recall=recall,
                                                                                                                f1=f1))
        else:
            statistics.append(
                "{key}: Precision：{precision:.4f}; Recall：{recall:.4f}; F1-score：{f1:.4f}".format(key='总计',
                                                                                                  precision=precision,
                                                                                                  recall=recall,
                                                                                                  f1=f1))
    return out, statistics
