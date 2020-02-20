from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import os
import sys

from tensorflow.python.framework.graph_util import convert_variables_to_constants
from seq2word_rnn_model import WordModel, LetterModel
from config import Config
import config
import data_feeder as data_feeder


FLAGS = config.FLAGS
data_type = config.data_type
index_data_type = config.index_data_type
np_index_data_type = config.np_index_data_type


def export_graph(session, iter, phase="lm"):
    if phase == "lm":
        # Export variables related to language model only
        variables_to_export = ["Online/WordModel/probabilities",
                               "Online/WordModel/state_out",
                               "Online/WordModel/top_k_prediction"]

    elif phase == "kc_full":
        # Export both language model and letter model's predictions
        variables_to_export = ["Online/WordModel/probabilities",
                               "Online/WordModel/state_out",
                               "Online/WordModel/top_k_prediction",
                               "Online/LetterModel/probabilities",
                               "Online/LetterModel/state_out",
                               "Online/LetterModel/top_k_prediction"]

    else:
        # Export language model's output and letter model's predictions, because we don't need language model's softmax.
        assert phase == "kc_slim"
        variables_to_export = ["Online/WordModel/state_out",
                               "Online/LetterModel/probabilities",
                               "Online/LetterModel/state_out",
                               "Online/LetterModel/top_k_prediction"]


    graph_def = convert_variables_to_constants(session, session.graph_def, variables_to_export)
    config_name = FLAGS.model_config
    model_export_path = os.path.join(FLAGS.graph_save_path)
    if not os.path.isdir(model_export_path):
        os.makedirs(model_export_path)
    model_export_name = os.path.join(model_export_path,
                                     config_name[config_name.rfind("/")+1:] + "-iter" + str(iter) + "-" + phase + '.pb')
    f = open(model_export_name, "wb")
    f.write(graph_def.SerializeToString())
    f.close()
    print("Graph is saved to: ", model_export_name)


def run_letter_epoch(session, data, word_model, letter_model, config, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    num_word = 0
    fetches = {}
    fetches_letter = {}
    previous_state = session.run(word_model.initial_state)

    for step, (epoch_size, lm_data, letter_data) in \
            enumerate(data_feeder.data_iterator(data, config)):
        if FLAGS.laptop_discount > 0 and step >= FLAGS.laptop_discount:
            break
        if step >= epoch_size:
            break

        fetches["rnn_state"] = word_model.rnn_state
        fetches["final_state"] = word_model.final_state
        fetches_letter["cost"] = letter_model.cost

        if eval_op is not None:
            fetches_letter["eval_op"] = eval_op
        feed_dict = {word_model.input_data: lm_data[0],
                     word_model.target_data: lm_data[1],
                     word_model.output_masks: lm_data[2],
                     word_model.sequence_length: lm_data[3],
                     word_model.initial_state: previous_state}
        # The language model's final states of previous epoch is the language model's initial state of current epoch.

        vals = session.run(fetches, feed_dict)

        previous_state = vals["final_state"]
        rnn_state_to_letter_model = vals["rnn_state"]

        feed_dict_letter = {letter_model.lm_state_in: rnn_state_to_letter_model,
                            letter_model.input_data: letter_data[0],
                            letter_model.target_data: letter_data[1],
                            letter_model.output_masks: letter_data[2],
                            letter_model.sequence_length: letter_data[3]}
        # The language model's rnn states of current epoch is the letter model's initial state of current epoch.

        vals_letter = session.run(fetches_letter, feed_dict_letter)
        cost_letter = vals_letter["cost"]

        costs += cost_letter
        iters += np.sum(letter_data[2])
        num_word += np.sum(letter_data[3])

        if verbose and step % (epoch_size // 1) == 0:
            if costs / iters > 100.0:
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] PPL TOO LARGE! %.3f ENTROPY: (%.3f) speed: %.0f wps" %
                      (step * 1.0 / epoch_size, costs / iters, num_word / (time.time() - start_time)))
            else:
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] %.3f letter_ppl: %.3f speed: %.0f wps" %
                      (step * 1.0 / epoch_size, np.exp(costs / iters), num_word / (time.time() - start_time)))
            sys.stdout.flush()

    return np.exp(costs / iters)


def run_word_epoch(session, data, word_model, config, lm_phase_id, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0

    num_word = 0
    fetches = {}
    previous_state = session.run(word_model.initial_state)

    for step, (epoch_size, lm_data, _) in \
            enumerate(data_feeder.data_iterator(data, config)):
        if FLAGS.laptop_discount > 0 and step >= FLAGS.laptop_discount:
            break
        if step >= epoch_size:
            break

        fetches["cost"] = word_model.cost
        fetches["final_state"] = word_model.final_state

        if eval_op is not None:
            fetches["eval_op"] = eval_op

        feed_dict = {word_model.input_data: lm_data[0],
                     word_model.target_data: lm_data[1],
                     word_model.output_masks: lm_data[2],
                     word_model.sequence_length: lm_data[3],
                     word_model.initial_state: previous_state
                     }
        # The language model's final states of previous epoch is the language model's initial state of current epoch.

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        previous_state = vals["final_state"]

        costs += cost
        iters += np.sum(lm_data[2])

        num_word += np.sum(lm_data[3])
        if verbose and step % (epoch_size // 1) == 0:
            if lm_phase_id == 0:
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] "
                      "%.3f word ppl: %.3f speed: %.0f wps"
                      % (step * 1.0 / epoch_size, np.exp(costs / iters),
                         num_word / (time.time() - start_time)))

            sys.stdout.flush()
    costs_list = np.exp(costs / iters), 1, 1
    return costs_list[lm_phase_id]


def run_test_epoch(session, data, word_model, letter_model, config, logfile, phase="lm"):
    """Runs the model on the given data."""
    start_time = time.time()
    fetches = {}
    fetches_letter = {}
    previous_state = session.run(word_model.initial_state)

    batch_size = config.batch_size
    num_steps = config.num_steps
    letter_batch_size = batch_size * num_steps

    prediction_made = 0.0
    top1_correct_total, top3_correct_total, top5_correct_total = 0.0, 0.0, 0.0
    top1_empty_correct_total, top3_empty_correct_total, top5_empty_correct_total = 0.0, 0.0, 0.0
    top1_complete_correct_total, top3_complete_correct_total, top5_complete_correct_total = 0.0, 0.0, 0.0

    for step, (epoch_size, lm_data, letter_data) in \
            enumerate(data_feeder.data_iterator(data, config)):
        if FLAGS.laptop_discount > 0 and step >= FLAGS.laptop_discount:
            break
        if step >= epoch_size:
            break

        fetches["rnn_state"] = word_model.rnn_state
        fetches["final_state"] = word_model.final_state
        fetches["top_k_prediction"] = word_model.top_k_prediction

        fetches_letter["top_k_prediction"] = letter_model.top_k_prediction

        feed_dict = {word_model.input_data: lm_data[0],
                     word_model.target_data: lm_data[1],
                     word_model.output_masks: lm_data[2],
                     word_model.sequence_length: lm_data[3],
                     word_model.initial_state: previous_state,
                     word_model.top_k: 5}
        # The language model's final states of previous epoch is the language model's initial state of current epoch.

        vals = session.run(fetches, feed_dict)

        previous_state = vals["final_state"]
        rnn_state_to_letter_model = vals["rnn_state"]

        feed_dict_letter = {letter_model.lm_state_in: rnn_state_to_letter_model,
                            letter_model.input_data: letter_data[0],
                            letter_model.target_data: letter_data[1],
                            letter_model.output_masks: letter_data[2],
                            letter_model.sequence_length: letter_data[3],
                            letter_model.top_k: 5}
        # The language model's rnn states of current epoch is the letter model's initial state of current epoch.

        vals_letter = session.run(fetches_letter, feed_dict_letter)
        if phase == "lm":
            top_k_prediction = vals["top_k_prediction"]
            # language model top_k_prediction.shape = [batch_size * num_steps, 5]
            y_as_a_column = lm_data[1].reshape([-1])
            mask_as_a_column = lm_data[2].reshape([-1])
            prediction_made += np.sum(mask_as_a_column)
            # sum of masks

            top1_correct = np.sum((top_k_prediction[:, 0] == y_as_a_column) * mask_as_a_column)
            top3_correct = top1_correct + np.sum((top_k_prediction[:, 1] == y_as_a_column) * mask_as_a_column) \
                           + np.sum((top_k_prediction[:, 2] == y_as_a_column) * mask_as_a_column)
            top5_correct = top3_correct + np.sum((top_k_prediction[:, 3] == y_as_a_column) * mask_as_a_column) \
                           + np.sum((top_k_prediction[:, 4] == y_as_a_column) * mask_as_a_column)
            top1_correct_total += top1_correct
            top3_correct_total += top3_correct
            top5_correct_total += top5_correct
        else:
            top_k_prediction = vals_letter["top_k_prediction"]
            # letter model top_k_prediction.shape = [batch_size * num_steps * max_word_length, 5]
            # for each row in y, extract only one label.
            # letter_data[1].shape = [batch_size * num_steps, max_word_length]
            y_as_a_column = letter_data[1][:, -1]
            letter_lengths = letter_data[3]
            # letter_length.shape = [batch_size * num_steps]
            prediction_made += letter_batch_size  # num of complete rows

            # rows where <start> flags occurs in the letter sequence
            empty_rows = np.arange(letter_batch_size) * config.max_word_length
            top_k_ids_empty_only = top_k_prediction[empty_rows, :]

            top1_empty_correct = np.sum((top_k_ids_empty_only[:, 0] == y_as_a_column))
            top3_empty_correct = top1_empty_correct + np.sum((top_k_ids_empty_only[:, 1] == y_as_a_column)) \
                                 + np.sum((top_k_ids_empty_only[:, 2] == y_as_a_column))
            top5_empty_correct = top3_empty_correct + np.sum((top_k_ids_empty_only[:, 3] == y_as_a_column)) \
                                 + np.sum((top_k_ids_empty_only[:, 4] == y_as_a_column))

            # rows where letters of current word are complete
            # letter_lengths = true length + 1 (prepended <start>)
            complete_rows = np.arange(letter_batch_size) * config.max_word_length + letter_lengths - 1
            top_k_ids_complete_only = top_k_prediction[complete_rows, :]

            top1_complete_correct = np.sum((top_k_ids_complete_only[:, 0] == y_as_a_column))
            top3_complete_correct = top1_complete_correct + np.sum((top_k_ids_complete_only[:, 1] == y_as_a_column)) \
                                    + np.sum((top_k_ids_complete_only[:, 2] == y_as_a_column))
            top5_complete_correct = top3_complete_correct + np.sum((top_k_ids_complete_only[:, 3] == y_as_a_column)) \
                                    + np.sum((top_k_ids_complete_only[:, 4] == y_as_a_column))

            top1_empty_correct_total += top1_empty_correct
            top3_empty_correct_total += top3_empty_correct
            top5_empty_correct_total += top5_empty_correct

            top1_complete_correct_total += top1_complete_correct
            top3_complete_correct_total += top3_complete_correct
            top5_complete_correct_total += top5_complete_correct

    if phase == "lm":
        top1_acc = top1_correct_total / prediction_made
        top3_acc = top3_correct_total / prediction_made
        top5_acc = top5_correct_total / prediction_made
        # Prediction accuracy information:
        print("Language model: Top1 accuracy = {0}, top3 accuracy = {1}, top5 accuracy = {2} ".format(top1_acc, top3_acc, top5_acc),
              file=logfile)
    else:
        top1_acc = top1_empty_correct_total / prediction_made
        top3_acc = top3_empty_correct_total / prediction_made
        top5_acc = top5_empty_correct_total / prediction_made
        # Prediction accuracy information:
        print("Accuracy with letter start: top1 {0}, top3 = {1}, top5 = {2} ".format(top1_acc, top3_acc, top5_acc),
              file=logfile)

        top1_acc = top1_complete_correct_total / prediction_made
        top3_acc = top3_complete_correct_total / prediction_made
        top5_acc = top5_complete_correct_total / prediction_made
        # Prediction accuracy information:
        print(
            "Accuracy with full letters: top1 {0}, top3 = {1}, top5 = {2} ".format(top1_acc, top3_acc, top5_acc),
            file=logfile)
    end_time = time.time()
    print("Test time = {0}".format(end_time - start_time), file=logfile)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data directory")

    logfile = open(FLAGS.model_config + '.log', 'w')

    config = Config()
    config.get_config(FLAGS.vocab_path, FLAGS.model_config)

    test_config = Config()
    test_config.get_config(FLAGS.vocab_path, FLAGS.model_config)
    test_config.batch_size = 1
    test_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_fraction
        train_data = data_feeder.read_file(FLAGS.data_path, config, is_train=True)
        valid_data = data_feeder.read_file(FLAGS.data_path, config, is_train=False)
        print("in words vocabulary size = %d\nout words vocabulary size = %d\nin letters vocabulary size = %d"%(config.vocab_size_in, config.vocab_size_out, config.vocab_size_letter))

        with tf.Session(config=gpu_config) as session:
            with tf.name_scope("Train"):
                # train model on train data
                with tf.variable_scope("WordModel", reuse=False, initializer=initializer):
                    mtrain_word = WordModel(is_training=True, config=config)
                    train_word_op = mtrain_word.train_op
                with tf.variable_scope("LetterModel", reuse=False, initializer=initializer):
                    mtrain_letter = LetterModel(is_training=True, config=config)
                    train_letter_op = mtrain_letter.train_op

            with tf.name_scope("Valid"):
                # valid model on valid data
                with tf.variable_scope("WordModel", reuse=True, initializer=initializer):
                    mvalid_word = WordModel(is_training=False, config=config)
                with tf.variable_scope("LetterModel", reuse=True, initializer=initializer):
                    mvalid_letter = LetterModel(is_training=False, config=config)

            with tf.name_scope("Test"):
                # test model on test data
                with tf.variable_scope("WordModel", reuse=True, initializer=initializer):
                    mtest_word = WordModel(is_training=False, config=config)
                with tf.variable_scope("LetterModel", reuse=True, initializer=initializer):
                    mtest_letter = LetterModel(is_training=False, config=config)

            with tf.name_scope("Online"):
                # language model and letter model to be saved and exported, and the batch size is 1.
                with tf.variable_scope("WordModel", reuse=True, initializer=initializer):
                    monline_word = WordModel(is_training=False, config=test_config)
                with tf.variable_scope("LetterModel", reuse=True, initializer=initializer):
                    monline_letter = LetterModel(is_training=False, config=test_config)

            restore_variables = dict()
            for v in tf.trainable_variables():
                print("store:", v.name)
                restore_variables[v.name] = v
            sv = tf.train.Saver(restore_variables)

            if not FLAGS.model_name.endswith(".ckpt"):
                FLAGS.model_name += ".ckpt"

            session.run(tf.global_variables_initializer())
            
            check_point_dir = os.path.join(FLAGS.save_path)
            ckpt = tf.train.get_checkpoint_state(check_point_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                sv.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")

            save_path = os.path.join(FLAGS.save_path)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            print("training language model.")
            print("training language model", file=logfile)
            # Train language model.

            # if FLAGS.use_phrase:
            #     lm_phase_num = 3
            # else:
            lm_phase_num = 1
            for lm_phase_id in range(lm_phase_num):
                print("lm training phase: %d" % (lm_phase_id + 1), file=logfile)
                if lm_phase_id != 1:
                    max_train_epoch = config.max_max_epoch
                else:
                    max_train_epoch = 2

                for i in range(max_train_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
                    print("lm training phase: %d" % (lm_phase_id + 1))
                    mtrain_word.assign_lr(session, config.learning_rate * lr_decay)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain_word.lr)), file=logfile)
                    train_perplexity = run_word_epoch(session, train_data, mtrain_word, config, lm_phase_id,
                                                      train_word_op, verbose=True)

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity), file=logfile)
                    logfile.flush()

                    valid_perplexity = run_word_epoch(session, valid_data, mvalid_word, config, lm_phase_id)
                    run_test_epoch(session, valid_data, mtest_word, mtest_letter, config,
                                                     logfile, phase="lm")

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Valid Perplexity: %.3f" %
                          (i + 1, valid_perplexity), file=logfile)
                    logfile.flush()

                    if FLAGS.save_path:
                        print("Saving model to %s." % FLAGS.save_path + "\n\n", file=logfile)
                        step = mtrain_word.get_global_step(session)
                        model_save_path = os.path.join(save_path, FLAGS.model_name)
                        sv.save(session, model_save_path, global_step=step)
                    print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin exporting lm graph!")
                    export_graph(session, i, phase="lm")
                    # Export dense graph
                    print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish exporting lm graph!")
                
            print("training letter model.")
            print("training letter model", file=logfile)
            # Train letter model.
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)

                mtrain_letter.assign_lr(session, config.learning_rate * lr_decay)
                print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain_letter.lr)), file=logfile)
                train_perplexity = run_letter_epoch(session, train_data, mtrain_word, mtrain_letter, config,
                                                    train_letter_op, verbose=True)
                print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                print("Epoch: %d Train ppl: %.3f" % (i + 1, train_perplexity), file=logfile)
                logfile.flush()

                print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                valid_perplexity = run_letter_epoch(session, valid_data, mtrain_word, mvalid_letter, config)
                run_test_epoch(session, valid_data, mtest_word, mtest_letter, config,
                                                 logfile, phase="letter")

                print("Epoch: %d Valid Perplexity: %.3f" %
                      (i + 1, valid_perplexity), file=logfile)
                logfile.flush()

                print("Saving model to %s." % FLAGS.save_path + "\n\n", file=logfile)
                step = mtrain_letter.get_global_step(session)
                model_save_path = os.path.join(save_path, FLAGS.model_name)
                sv.save(session, model_save_path, global_step=step)
                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin exporting letter model graph!")

                export_graph(session, i, phase="kc_full")
                export_graph(session, i, phase="kc_slim")
                # Export dense graph

                print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish exporting letter model graph!")
            
            logfile.close()


if __name__ == "__main__":
    tf.app.run()
