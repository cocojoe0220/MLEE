# -*- coding: utf-8 -*-
"""Extract pre-computed feature vectors from BERT."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .tokenization import FullTokenizer, validate_case_matches_checkpoint, BasicTokenizer, printable_text
from .fast_modeling import BertConfig
from tensorflow.python.estimator.estimator import Estimator
from .fast_modeling import BertModel, get_assignment_map_from_checkpoint
import tensorflow as tf

import threading
import numpy as np
from container import bert_tiny_dict, feature_dict_path
import json


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


def input_fn(text):
    examples = []
    examples.append(
        InputExample(unique_id=0, text_a=text, text_b=None))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % example.unique_id)
            tf.logging.info("tokens: %s" % " ".join(
                [printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class Fast(object):
    def __init__(self):
        self.closed = False
        self.first_run = True
        self.tokenizer = FullTokenizer(
            vocab_file=bert_tiny_dict['vocab_file'],
            do_lower_case=True)
        self.layers = bert_tiny_dict['layers']
        self.init_checkpoint = bert_tiny_dict['init_checkpoint']
        self.max_seq_length = bert_tiny_dict['max_seq_length']
        self.bert_config_file = bert_tiny_dict['bert_config_file']
        self.use_tpu = bert_tiny_dict['use_tpu']
        self.use_one_hot_embeddings = bert_tiny_dict['use_one_hot_embeddings']
        self.text = None
        self.num_examples = None
        # self.predictions = None
        self.result = None
        self.estimator = self.get_estimator()
        self.lock = threading.Lock()

    def model_fn_builder(self, bert_config, init_checkpoint, layer_indexes, use_tpu,
                         use_one_hot_embeddings):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            unique_ids = features["unique_ids"]
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            input_type_ids = features["input_type_ids"]
            tokens = features["tokens"]

            model = BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)

            if mode != tf.estimator.ModeKeys.PREDICT:
                raise ValueError("Only PREDICT modes are supported: %s" % (mode))

            tvars = tf.trainable_variables()
            scaffold_fn = None
            (assignment_map,
             initialized_variable_names) = get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)
            all_layers = model.get_all_encoder_layers()

            predictions = {
                "unique_id": unique_ids,
                "tokens": tokens
            }

            for (i, layer_index) in enumerate(layer_indexes):
                predictions["layer_output_%d" % i] = all_layers[layer_index]
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
            return output_spec

        return model_fn

    def get_estimator(self):
        validate_case_matches_checkpoint(True, self.init_checkpoint)
        bert_config = BertConfig.from_json_file(self.bert_config_file)  # 载入bert自定义配置
        layer_indexes = [int(x) for x in self.layers.split(",")]
        model_fn = self.model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=self.init_checkpoint,
            layer_indexes=layer_indexes,
            use_tpu=self.use_tpu,
            use_one_hot_embeddings=self.use_one_hot_embeddings)

        session_config = tf.ConfigProto(log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        run_config = tf.estimator.RunConfig().replace(session_config=session_config)

        estimator = Estimator(  # 实例化估计器
            model_fn=model_fn,
            config=run_config
        )

        return estimator

    def get_feature(self, index, text):
        examples = input_fn(text)
        features = convert_examples_to_features(examples=examples, seq_length=self.max_seq_length,
                                                tokenizer=self.tokenizer)
        return features[0].unique_id, features[0].input_ids, features[0].input_mask, features[0].input_type_ids, \
               features[0].tokens

    def create_generator(self):
        """构建生成器"""
        while not self.closed:
            self.num_examples = len(self.text)
            features = (self.get_feature(*f) for f in enumerate(self.text))
            yield dict(zip(("unique_ids", "input_ids", "input_mask", "input_type_ids", "tokens"), zip(*features)))

    def input_fn_builder(self):
        """用于预测单独对预测数据进行创建，不基于文件数据"""

        dataset = tf.data.Dataset.from_generator(
            self.create_generator,
            output_types={'unique_ids': tf.int32,
                          'input_ids': tf.int32,
                          'input_mask': tf.int32,
                          'input_type_ids': tf.int32,
                          'tokens': tf.string},
            output_shapes={'unique_ids': (None),
                           'tokens': (None),
                           'input_ids': (None, None),
                           'input_mask': (None, None),
                           'input_type_ids': (None, None)}
        )
        return dataset

    def predict(self, text):
        self.text = text
        tokens_list = []
        if self.first_run:
            self.result = self.estimator.predict(input_fn=self.input_fn_builder, yield_single_examples=True)
            self.first_run = False
        for i, t in enumerate(text):
            with self.lock:
                result = next(self.result)
                tokens_list.append([x.decode('utf-8') for x in list(result['tokens'])])
                result = np.expand_dims(result['layer_output_0'], 0)
                if i == 0:
                    results = result
                else:
                    results = np.concatenate([results, result], 0)
        return results, tokens_list

    def close(self):
        self.closed = True


fast_bert_train = Fast()

with open(feature_dict_path, encoding='utf-8') as f:
    features_dict = json.load(f)
