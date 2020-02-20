from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import config

FLAGS = config.FLAGS
data_type = config.data_type
index_data_type = config.index_data_type
np_index_data_type = config.np_index_data_type


class WordModel(object):
    # Below is the language model.

    def __init__(self, is_training, config):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.embedding_size = config.word_embedding_size
        self.hidden_size = config.word_hidden_size
        self.vocab_size_in = config.vocab_size_in
        self.vocab_size_out = config.vocab_size_out


        self.input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                         name="batched_input_word_ids")
        self.target_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                          name="batched_output_word_ids")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_output_word_masks")
        self.sequence_length = tf.placeholder_with_default(input=tf.fill(dims=[self.batch_size], value=self.num_steps),
                                                           shape=[self.batch_size], name="batched_input_sequence_length")
        self.top_k = tf.placeholder(dtype=index_data_type(), shape=[], name="top_k")


        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        lstm_state_as_tensor_shape = [config.num_layers, 2, config.batch_size, config.word_hidden_size]
        
        self._initial_state = tf.placeholder_with_default(tf.zeros(lstm_state_as_tensor_shape, dtype=data_type()),
                                                          lstm_state_as_tensor_shape, name="state")

        unstack_state = tf.unstack(self._initial_state, axis=0)
        tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1])
             for idx in range(config.num_layers)]
        )

        with tf.variable_scope("Lm"):
            with tf.variable_scope("Embedding"):
                self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.embedding_size],
                                                  dtype=data_type())
                inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)

                # inputs is of shape [batch_size, num_steps, word_embedding_size]

                embedding_to_rnn = tf.get_variable("embedding_to_rnn", [self.embedding_size, self.hidden_size],
                                                   dtype=data_type())
                inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]), embedding_to_rnn),
                                    shape=[self.batch_size, -1, self.hidden_size])

                # Now inputs is of shape [batch_size, num_steps, word_hidden_size]

                if is_training and config.keep_prob < 1:
                    inputs = tf.nn.dropout(inputs, config.keep_prob)

            with tf.variable_scope("RNN"):
                outputs = list()
                states = list()
                state = tuple_state

                for timestep in range(self.num_steps):

                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output, state) = cell(inputs[:, timestep, :], state)
                    outputs.append(output)
                    states.append(state)

                rnn_output = tf.transpose(outputs, perm=[1, 0, 2])
                # rnn_output is a Tensor of shape [batch_size, num_steps, word_hidden_size]

                rnn_output = tf.reshape(rnn_output, [-1, self.hidden_size])
                # Now rnn_output is a Tensor of shape [batch_size * num_steps, word_hidden_size]

                states = tf.transpose(states, perm=[3, 1, 2, 0, 4])
                # states is a Tensor of shape [batch_size, num_layers, 2, num_steps, word_hidden_size]

                unstack_states = tf.unstack(states, axis=0)
                rnn_state = tf.concat(unstack_states, axis=2)
                # Now rnn_state is a Tensor of shape [num_layers, 2, batch_size * num_steps, word_hidden_size]

            with tf.variable_scope("Softmax"):
                rnn_output_to_final_output = tf.get_variable("rnn_output_to_final_output",
                                                             [self.hidden_size, self.embedding_size],
                                                             dtype=data_type())
                self._softmax_w = tf.get_variable("softmax_w", [self.embedding_size, self.vocab_size_out],
                                                  dtype=data_type())
                softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=data_type())


        logits = tf.matmul(tf.matmul(rnn_output, rnn_output_to_final_output), self._softmax_w) + softmax_b

        probabilities = tf.nn.softmax(logits, name="probabilities")
        # probabilities.shape = [batch_size * num_steps, vocab_size_out]

        _, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.target_data, [-1])],
                                                                  [tf.reshape(self.output_masks, [-1])],
                                                                  average_across_timesteps=False)

        self._cost = cost = tf.reduce_sum(loss)

        self._final_state = tf.identity(state, "state_out")
        self._rnn_state = tf.identity(rnn_state, "rnn_state")
        self._logits = logits
        self._probabilities = probabilities
        self._top_k_prediction = top_k_prediction

        if not is_training:
            return

        self._lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(config.learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/Lm")

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def get_global_step(self, session):
        gs = session.run(self.global_step)
        return gs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def softmax_w(self):
        return self._softmax_w

    @property
    def cost(self):
        return self._cost

    @property
    def embedding(self):
        return self._embedding

    @property
    def final_state(self):
        return self._final_state

    @property
    def rnn_state(self):
        return self._rnn_state

    @property
    def lr(self):
        return self._lr

    @property
    def logits(self):
        return self._logits

    @property
    def probalities(self):
        return self._probabilities

    @property
    def top_k_prediction(self):
        return self._top_k_prediction

    @property
    def train_op(self):
        return self._train_op


class LetterModel(object):
    # Below is the letter model.

    def __init__(self, is_training, config):
        self.num_steps = config.num_steps
        self.batch_size = config.batch_size * config.num_steps
        # the batch_size of letter model equals to batch_size * num_steps

        self.max_word_length = config.max_word_length
        self.embedding_size = config.letter_embedding_size
        self.hidden_size = config.letter_hidden_size
        self.vocab_size_in = config.vocab_size_letter
        self.vocab_size_out = config.vocab_size_out

        self.input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                         name="batched_input_word_ids")
        self.target_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                          name="batched_output_word_ids")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_output_word_masks")
        self.sequence_length = tf.placeholder_with_default(input=tf.fill(dims=[self.batch_size],
                                                           value=self.max_word_length),
                                                           shape=[self.batch_size],
                                                           name="batched_input_sequence_length")
        self.top_k = tf.placeholder(dtype=index_data_type(), shape=[], name="top_k")

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        lm_state_as_tensor_shape = [config.num_layers, 2, self.batch_size, config.word_hidden_size]
        letter_state_as_tensor_shape = [config.num_layers, 2, self.batch_size, config.letter_hidden_size]

        self.lm_state_in = tf.placeholder_with_default(tf.zeros(lm_state_as_tensor_shape, dtype=data_type()),
                                                       lm_state_as_tensor_shape, name="lm_state_in")
        # lm_state_in is of shape [num_layers, 2, batch_size * num_steps, word_hidden_size]

        with tf.variable_scope("StateMatrix"):

            lm_state_to_letter_state = tf.get_variable("lm_state_to_letter_state",
                                                       [config.word_hidden_size, config.letter_hidden_size],
                                                       dtype=data_type())

        if config.word_hidden_size != config.letter_hidden_size:
            self._initial_state = tf.placeholder_with_default(
                                tf.reshape(tf.matmul(tf.reshape(self.lm_state_in, [-1, config.word_hidden_size]),
                                                     lm_state_to_letter_state), letter_state_as_tensor_shape),
                                                     letter_state_as_tensor_shape, name="state")
        else:
            self._initial_state = tf.placeholder_with_default(
                self.lm_state_in, letter_state_as_tensor_shape, name="state")

        # initial_state is of shape [num_layers, 2, batch_size * num_steps, letter_hidden_size]

        unstack_state = tf.unstack(self._initial_state, axis=0)
        tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1])
             for idx in range(config.num_layers)]
        )

        with tf.variable_scope("Embedding"):

            self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.embedding_size], dtype=data_type())

            inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)
            # inputs is of shape [batch_size * num_steps, max_word_length, letter_embedding_size]

            embedding_to_rnn = tf.get_variable("embedding_to_rnn",
                                               [self.embedding_size, self.hidden_size],
                                               dtype=data_type())
            inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]), embedding_to_rnn),
                                shape=[self.batch_size, -1, self.hidden_size])
            # now inputs is of shape [batch_size * num_steps, max_word_length, letter_hidden_size]

            if is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

        with tf.variable_scope("RNN"):
            outputs, state_out = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.sequence_length,
                                                   initial_state=tuple_state)
            # outputs is a Tensor of shape [batch_size * num_steps, max_word_length, letter_hidden_size]
            # state_out is a tuple of tuple of Tensor: state_out = ((c1, h1), (c2, h2), ..., (cl, hl))

        output = tf.reshape(outputs, [-1, self.hidden_size])
        # Now output is a Tensor of shape [batch_size * num_steps * max_word_length, letter_hidden_size]
        with tf.variable_scope("Softmax"):
            rnn_output_to_final_output = tf.get_variable("rnn_output_to_final_output",
                                                         [self.hidden_size, self.embedding_size],
                                                         dtype=data_type())
            self._softmax_w = tf.get_variable("softmax_w", [self.embedding_size, self.vocab_size_out],
                                              dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=data_type())

        logits = tf.matmul(tf.matmul(output, rnn_output_to_final_output),
                           self._softmax_w) + softmax_b

        probabilities = tf.nn.softmax(logits, name="probabilities")
        # probabilities.shape = [batch_size * num_steps * max_word_length, vocab_size_out]
        _, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.target_data, [-1])],
                                                                  [tf.reshape(self.output_masks, [-1])],
                                                                  average_across_timesteps=False)

        self._cost = cost = tf.reduce_sum(loss)

        self._final_state = tf.identity(state_out, "state_out")
        self._logits = logits
        self._probabilities = probabilities
        self._top_k_prediction = top_k_prediction

        if not is_training:
            return

        self._lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(config.learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LetterModel")

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def get_global_step(self, session):
        gs = session.run(self.global_step)
        return gs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def softmax_w(self):
        return self._softmax_w

    @property
    def cost(self):
        return self._cost

    @property
    def embedding(self):
        return self._embedding

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def logits(self):
        return self._logits

    @property
    def probalities(self):
        return self._probabilities

    @property
    def top_k_prediction(self):
        return self._top_k_prediction

    @property
    def train_op(self):
        return self._train_op


