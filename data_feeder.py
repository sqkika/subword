from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import codecs


def read_lm_data(data_path, num_steps):
    # Load data file named "train_in_ids_lm" or "dev_in_ids_lm" in the data_path.
    lm_in_ids_list, lm_out_ids_list = [], []

    with codecs.open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            lm_in, lm_out = line.strip().split("#")
            # lm_in and lm_out are the in_vocab_ids and out_vocab_ids for one sentence, separated by "#".
            # e.g.: 0 1 2 3#0 1 2 3
            lm_in_ids = lm_in.split()[:num_steps]
            lm_out_ids = lm_out.split()[:num_steps]
            lm_in_ids_list.append(lm_in_ids)
            lm_out_ids_list.append(lm_out_ids)

    return lm_in_ids_list, lm_out_ids_list


def read_letter_data(data_path, num_steps, max_word_length):
    # Load data file named "train_in_ids_letters" or "dev_in_ids_letters" in the data_path.
    letter_ids_list = []
    letter_length_list = []

    with codecs.open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            letters = line.strip().split("#")[:num_steps]
            # letters are the letter ids of every word in one sentence, separated by "#".
            # e.g.: 1 2 3#1 3 2 4...
            for letter_ids in letters:
                letter_ids_split = letter_ids.split()[:max_word_length]
                letter_ids_list.append(letter_ids_split + [0]*(max_word_length - len(letter_ids_split)))
                letter_length_list.append(len(letter_ids_split))

    return letter_ids_list, letter_length_list


def read_file(data_path, config, is_train=False):

    mode = "train" if is_train else "dev"

    lm_data_file = os.path.join(data_path, mode + "_in_ids_lm")
    letter_file = os.path.join(data_path, mode + "_in_ids_letters")

    head_mask = 0

    lm_in_data, lm_out_data = read_lm_data(lm_data_file, config.num_steps)
    # lm_in_data is a list of lists which represent the word ids in the in_vocab of every sentence.
    # lm_out_data is a list of lists which represent the word ids in the out_vocab of every sentence.

    letter_data, letter_length = read_letter_data(letter_file, config.num_steps, config.max_word_length)
    # letter_data is a list of lists which represent the letter ids of every word.


    # phrase_data is a list of lists which represent the phrase ids of every sentence.

    assert len(lm_in_data) == len(lm_out_data)
    print(mode + " data size: ", len(lm_in_data))

    return [lm_in_data, lm_out_data, letter_data, letter_length, head_mask]
    # return [lm_in_data, lm_out_data]

def data_iterator(data, config):

    lm_unuesd_num = 2
    phrase_unused_num = 2

    lm_in_data = data[0]
    lm_out_data = data[1]
    letter_data = data[2]
    letter_length = data[3]
    head_mask = data[4]

    num_steps = config.num_steps
    max_word_length = config.max_word_length
    batch_size = config.batch_size

    def flatten(lst):
        # Assume lst is a list of lists, whose every item is an ID list
        # This function flatten lst into a list of IDs.
        # e.g.: [[1, 2], [3, 4, 5, 6], [7, 8, 9]] -> [1, 2, 3, 4, 5, 6, 7, 8, 9]
        return [x for item in lst for x in item]

    def maskWeight(letter_num, letter, out_data):

        if letter_num == 1:
            return 10.0
        return 15
        # # when letter_num is 1(only letter start id), it means it is a emoji, and emoji mask is 10.
        # in_letters_id = letter[1: letter_num]
        # print(in_letters_id)
        # print(config.data_utility.id2token_in_letters)
        # in_letters = [config.data_utility.id2token_in_letters[int(id)] for id in in_letters_id]
        # in_word = ''.join(in_letters)
        # out_word = config.data_utility.id2token_out[int(out_data)]
        # return 15.0 if in_word == out_word else 5.0
        # The mask is 15 when the input letter equals to the output word, else the mask is 5.

    while True:

        lm_in_epoch = flatten(lm_in_data)
        lm_out_epoch = flatten(lm_out_data)

        batch_length = len(lm_in_epoch) // batch_size
        valid_epoch_range = batch_size * batch_length

        lm_in_epoch = np.reshape(np.array(lm_in_epoch[:valid_epoch_range], dtype=np.int32), [batch_size, -1])
        lm_out_epoch = np.reshape(np.array(lm_out_epoch[:valid_epoch_range], dtype=np.int32), [batch_size, -1])


        letter_epoch = np.reshape(np.array(letter_data[:valid_epoch_range], dtype=np.int32), [batch_size, -1, max_word_length])
        letter_length = np.reshape(np.array(letter_length[:valid_epoch_range], dtype=np.int32), [batch_size, -1])



        epoch_size = (batch_length - 1) // num_steps

        for i in range(epoch_size):

            lm_epoch_x = lm_in_epoch[:, i * num_steps:(i + 1) * num_steps]
            lm_epoch_y = lm_out_epoch[:, i * num_steps + 1:(i + 1) * num_steps + 1]

            lm_epoch_y_as_a_column = lm_epoch_y.reshape([-1])


            letter_epoch_x = np.reshape(letter_epoch[:, i * num_steps + 1:(i + 1) * num_steps + 1, :],
                                        [-1, max_word_length])
            letter_epoch_y = np.repeat(lm_epoch_y, max_word_length).reshape([-1, max_word_length])
            letter_length_epoch = np.reshape(letter_length[:, i * num_steps + 1:(i + 1) * num_steps + 1], [-1])

            letter_mask_epoch = np.array([[1.0] * (length - 1) + [maskWeight(length, letter, word)]
                                         + [0.0] * (max_word_length - length) if length > 0 else
                                         [0.0] * max_word_length for (letter, length, word) in
                                         zip(letter_epoch_x, letter_length_epoch, lm_epoch_y_as_a_column)])

            unused_letter_mask = (lm_epoch_y < lm_unuesd_num).reshape([-1])
            letter_mask_epoch[unused_letter_mask == True] = 0.0
            # Do not calculate loss for <eos> position in the letter epoch.


            sequence_lengths = np.array([num_steps] * batch_size, dtype=np.int32)

            lm_mask = np.ones([batch_size, num_steps])
            lm_mask[lm_epoch_y < lm_unuesd_num] = 0.0
            # Do not calculate loss for <eos> position


            # Do not calculate loss for phrase prob pad position
            # Do not calculate loss for phrase pad position

            data_feed_to_lm_model = (lm_epoch_x, lm_epoch_y, lm_mask, sequence_lengths)


            data_feed_to_letter_model = (letter_epoch_x, letter_epoch_y, letter_mask_epoch, letter_length_epoch)

            # yield epoch_size, data_feed_to_lm_model
            yield epoch_size, data_feed_to_lm_model, data_feed_to_letter_model
