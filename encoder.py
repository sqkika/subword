# coding=utf-8
""" An encoder which learns byte pair encodings for white-space separated text.  Can tokenize, encode, and decode. """
from collections import Counter
import re
try:
    from typing import Dict, Iterable, Callable, List, Any, Iterator
except ImportError:
    pass

from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
import toolz
import json

DEFAULT_EOW = '__eow'
DEFAULT_SOW = '__sow'
# DEFAULT_EOS = '__eos'
DEFAULT_UNK = '__unk'
DEFAULT_PAD = '__pad'


class Encoder:
    """ Encodes white-space separated text using byte-pair encoding.  See https://arxiv.org/abs/1508.07909 for details.
    """

    def __init__(self, vocab_size=8192, pct_bpe=0.2, word_tokenizer=None,
                 silent=True, ngram_min=2, ngram_max=2, required_tokens=None,
                 strict=False, lowercase=True,letter_path=None,
                 # EOW=DEFAULT_EOW, SOW=DEFAULT_SOW,EOS=DEFAULT_EOS,UNK=DEFAULT_UNK):
                 EOW=DEFAULT_EOW, SOW=DEFAULT_SOW,UNK=DEFAULT_UNK, PAD=DEFAULT_PAD):
        if vocab_size < 1:
            raise ValueError('vocab size must be greater than 0.')

        self.EOW = EOW
        self.SOW = SOW
        self.eow_len = len(EOW)
        self.sow_len = len(SOW)
        # self.EOS = EOS
        self.UNK = UNK
        self.PAD = PAD
        self.required_tokens = list(set(required_tokens or []).union({self.PAD,self.UNK,self.SOW,self.EOW}))
        # self.required_tokens = list(set(required_tokens or []).union({self.UNK, self.PAD}))
        # self.required_tokens = list(set(required_tokens or []).union({self.EOS,self.UNK}))
        self.vocab_size = vocab_size
        self.pct_bpe = pct_bpe
        self.word_vocab_size = max([int(vocab_size * (1 - pct_bpe)), len(self.required_tokens or [])])
        self.bpe_vocab_size = vocab_size - self.word_vocab_size
        self.word_tokenizer = word_tokenizer if word_tokenizer is not None else wordpunct_tokenize
        self.custom_tokenizer = word_tokenizer is not None
        self.word_vocab = {}  # type: Dict[str, int]
        self.bpe_vocab = {}  # type: Dict[str, int]
        self.inverse_word_vocab = {}  # type: Dict[int, str]
        self.inverse_bpe_vocab = {}  # type: Dict[int, str]
        self._progress_bar = iter if silent else tqdm
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.strict = strict
        self.lowercase = lowercase
        self.letter_path = letter_path
        self.letters_dict = {}
        self.letters = []

    def mute(self):
        """ Turn on silent mode """
        self._progress_bar = iter

    def unmute(self):
        """ Turn off silent mode """
        self._progress_bar = tqdm

    def byte_pair_counts(self, words):
        # type: (Encoder, Iterable[str]) -> Iterable[Counter]
        """ Counts space separated token character pairs:
            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4}
        """
        for token, count in self._progress_bar(self.count_tokens(words).items()):
            bp_counts = Counter()  # type: Counter
            for ngram in token.split(' '):
                bp_counts[ngram] += count
            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)]) + 1):
                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(ngram_size, token.split(' '))]

                for ngram in ngrams:
                    bp_counts[''.join(ngram)] += count

            yield bp_counts

    def count_tokens(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Count tokens into a BPE vocab """
        token_counts = Counter(self._progress_bar(words))
        return {' '.join(token): count for token, count in token_counts.items()}

    def learn_word_vocab(self, sentences):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Build vocab from self.word_vocab_size most common tokens in provided sentences """
        word_counts = Counter(word for word in toolz.concat(map(self.word_tokenizer, sentences)))
        for token in set(self.required_tokens or []):
            word_counts[self.PAD] = int(2**63)-1
            word_counts[self.UNK] = int(2**63)-2
            word_counts[self.SOW] = int(2**63)-3
            word_counts[self.EOW] = int(2**63)-4
        sorted_word_counts = sorted(word_counts.items(), key=lambda p: -p[1])
        return {word: idx for idx, (word, count) in enumerate(sorted_word_counts[:self.word_vocab_size])}

    def learn_bpe_vocab(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Learns a vocab of byte pair encodings """
        vocab = Counter()  # type: Counter
        # for token in {self.SOW, self.EOW}:
        #     vocab[token] = int(2**63)
        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):
            for byte_pair, count in byte_pair_count.items():
                vocab[byte_pair] += count

            if (idx + 1) % 10000 == 0:
                self.trim_vocab(10 * self.bpe_vocab_size, vocab)

        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:self.bpe_vocab_size]
        return {bp: idx + self.word_vocab_size for idx, (bp, count) in enumerate(sorted_bpe_counts)}

    def create_lettervocab(self,text):
        # type: (Encoder, Iterable[str]) -> None
        """ Learn vocab from text. """
        if self.lowercase:
            _text = [l.lower().strip() for l in text]
        else:
            _text = [l.strip() for l in text]
        # First, learn word vocab
        self.word_vocab = self.learn_word_vocab(_text)
        remaining_words = [word for word in toolz.concat(map(self.word_tokenizer, _text))
                   if word not in self.word_vocab]
        letters = {}
        for word in self.word_vocab:
            for letter in word:
                letters[letter] =1
        for word in remaining_words:
            for letter in word:
                letters[letter] =1
        with open(self.letter_path,'w') as f:
            i = 4
            for k,v in enumerate(letters):
                f.write(v+'##'+str(i)+'\n')
                i+=1

    def fit(self, text):
        # type: (Encoder, Iterable[str]) -> None
        """ Learn vocab from text. """
        if self.lowercase:
            _text = [l.lower().strip() for l in text]
        else:
            _text = [l.strip() for l in text]
        # First, learn word vocab
        self.word_vocab = self.learn_word_vocab(_text)
        # print(self.word_vocab)
        # self.letters = []
        # print(self.letter_path)
        with open(self.letter_path,'r') as f:
            for line in f.readlines():
                self.letters.append(line.split('##')[0])
        # print(self.letters)
        wrong = {}
        for k,v in enumerate(self.word_vocab):
            for letter in v:
                if letter not in self.letters:
                    wrong[v] = 1
                    continue
        # print(wrong)
        for k,v in enumerate(wrong):
            self.word_vocab.pop(v,-1)
        remaining_words = [word for word in toolz.concat(map(self.word_tokenizer, _text))
                           if word not in self.word_vocab]
        remainning_words_dict = {}
        for word in remaining_words:
            remainning_words_dict[word] = 1
        # print(remaining_words)
        remain_wrong = []
        for k,v in enumerate(remainning_words_dict):
            for letter in v:
                if letter not in self.letters:
                    # wrong[v] = 1
                    remain_wrong.append(v)
                    continue
 
        # print(remain_wrong)
        for word in remain_wrong:
            remainning_words_dict.pop(word,-1)
        remaining_words = [v for k,v in enumerate(remainning_words_dict)]
        # print(self.word_vocab)
        # print(remaining_words)
        self.bpe_vocab = self.learn_bpe_vocab(remaining_words)

        self.inverse_word_vocab = {idx: token for token, idx in self.word_vocab.items()}
        self.inverse_bpe_vocab = {idx: token for token, idx in self.bpe_vocab.items()}

    @staticmethod
    def trim_vocab(n, vocab):
        # type: (int, Dict[str, int]) -> None
        """  Deletes all pairs below 10 * vocab size to prevent memory problems """
        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
        pairs_to_trim = [pair for pair, count in pair_counts[n:]]
        for pair in pairs_to_trim:
            del vocab[pair]

    def subword_tokenize(self, word):
        # type: (Encoder, str) -> List[str]
        """ Tokenizes inside an unknown token using BPE """
        end_idx = min([len(word), self.ngram_max])
        sw_tokens = [self.SOW]
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self.bpe_vocab:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            elif len(subword) == 1:
                sw_tokens.append(self.UNK)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            else:
                end_idx -= 1

        sw_tokens.append(self.EOW)
        return sw_tokens

    def tokenize(self, sentence):
        # type: (Encoder, str) -> List[str]
        """ Split a sentence into word and subword tokens """
        if self.lowercase:
            word_tokens = self.word_tokenizer(sentence.lower().strip())
        else:
            word_tokens = self.word_tokenizer(sentence.strip())
        tokens = []
        numbers = ['0','1','2','3','4','5','6','7','8','9','+','-',
        '»','′','︿','⊙','—','﹏','←','☽','☾','≻','″','','√','|','°','⌣','¿','=',';','‘',
        '(',')','&','%','*','#','@','!','$','^',',','.','>','<','~']
        for word_token in word_tokens:
            unk = False
            for letter in word_token:
                if letter  in numbers:
                    unk=True
                    break
            if word_token in self.word_vocab:
                tokens.append(word_token)
            elif unk==False:
                tokens.extend(self.subword_tokenize(word_token))
            else:
                tokens.append(self.UNK)

        return tokens

    def transformletters(self, sub_sentences):
        self.letters_dict
        # print(self.letter_path)
        # with open(self.letter_path,'r') as f:
        #     for line in f.readlines():
        #         l,n = line.split('##')
        #         self.letters_dict[l] = n
        
        unk_pad = ['__pad','__unk']
        sentence_list = []
        len_list = []
        for k,subtoken in enumerate(sub_sentences):
            lenth = 0 
            if subtoken == '__sow':
                continue
                # sentence_list.append('1')
                # sentence_list.append(' ')
                # sentence_list.append('2')
                # lenth+=2
            elif subtoken == '__eow':
                continue
                # sentence_list.append('1')
                # sentence_list.append(' ')
                # sentence_list.append('3')
                # lenth+=2
            elif subtoken == '__unk':
                sentence_list.append('1')
                sentence_list.append(' ')
                sentence_list.append('0')
                lenth+=3
            elif subtoken == '__pad':
                sentence_list.append('1')
                sentence_list.append(' ')
                sentence_list.append('0')
                lenth+=3
            else:
                sentence_list.append('1')
                lenth+=1
                for i,letter in enumerate(subtoken):
                    ids = self.letters_dict[letter]
                    sentence_list.append(' ')
                    sentence_list.append(str(ids))
                    lenth+=2 
            if k <len(sub_sentences)-1:
                sentence_list.append('#')
            len_list.append(lenth)
        if sentence_list[-1]=='#':
            sentence_list = sentence_list[0:-1]
        sepcount = 0
        letters_line = '1 0#'
        # new_sentence_list = []
        # for i,ids in enumerate(sentence_list):
        #     if ids=='2':
        #         new_sentence_list.append(sentence_list[i-4])
        #     elif ids=='3':
        #         new_sentence_list.append(sentence_list[i-4])
        #         # pastlen = len_list[sepcount]
        #         # temp = sentence_list[i-pastlen-2:i-3]
        #         # for k in temp:                 
        #         #     new_sentence_list.append(k)
        #     else:
        #         new_sentence_list.append(ids)
        #         # if ids=='#':
        #         #     sepcount+=1
        for idx in sentence_list:
            letters_line+=idx
        return letters_line + '\n'

    def transform(self, sentences, reverse=False, fixed_length=None):
        # type: (Encoder, Iterable[str], bool, int) -> Iterable[List[int]]
        """ Turns space separated tokens into vocab idxs """
        direction = -1 if reverse else 1
        for sentence in self._progress_bar(sentences):
            in_subword = False
            encoded = []
            if self.lowercase:
                tokens = list(self.tokenize(sentence.lower().strip()))
            else:
                tokens = list(self.tokenize(sentence.strip()))
            for token in tokens:
                if in_subword:
                    if token in self.bpe_vocab:
                        if token == self.EOW:
                            in_subword = False
                        encoded.append(self.bpe_vocab[token])
                    else:
                        encoded.append(self.word_vocab[self.UNK])
                else:
                    if token == self.SOW:
                        in_subword = True
                        encoded.append(self.bpe_vocab[token])
                    else:
                        if token in self.word_vocab:
                            encoded.append(self.word_vocab[token])
                        else:
                            encoded.append(self.word_vocab[self.UNK])

            if fixed_length is not None:
                encoded = encoded[:fixed_length]
                while len(encoded) < fixed_length:
                    encoded.append(self.word_vocab[self.PAD])
            return encoded
            #yield encoded[::direction]

    def inverse_transform(self, rows):
        # type: (Encoder, Iterable[List[int]]) -> Iterator[str]
        """ Turns token indexes back into space-joined text. """
        for row in rows:
            words = []

            rebuilding_word = False
            current_word = ''
            for idx in row:
                if self.inverse_bpe_vocab.get(idx) == self.SOW:
                    if rebuilding_word and self.strict:
                        raise ValueError('Encountered second SOW token before EOW.')
                    rebuilding_word = True

                elif self.inverse_bpe_vocab.get(idx) == self.EOW:
                    if not rebuilding_word and self.strict:
                        raise ValueError('Encountered EOW without matching SOW.')
                    rebuilding_word = False
                    words.append(current_word)
                    current_word = ''

                elif rebuilding_word and (idx in self.inverse_bpe_vocab):
                    current_word += self.inverse_bpe_vocab[idx]

                elif rebuilding_word and (idx in self.inverse_word_vocab):
                    current_word += self.inverse_word_vocab[idx]

                elif idx in self.inverse_word_vocab:
                    words.append(self.inverse_word_vocab[idx])

                elif idx in self.inverse_bpe_vocab:
                    if self.strict:
                        raise ValueError("Found BPE index {} when not rebuilding word!".format(idx))
                    else:
                        words.append(self.inverse_bpe_vocab[idx])

                else:
                    raise ValueError("Got index {} that was not in word or BPE vocabs!".format(idx))

            yield ' '.join(w for w in words if w != '')

    def vocabs_to_dict(self, dont_warn=False):
        # type: (Encoder, bool) -> Dict[str, Dict[str, int]]
        """ Turns vocab into dict that is json-serializeable """
        print('before kick!')
        print(len(self.word_vocab))
        incorrect = []
        for k,v in self.word_vocab.items():
            if self.bpe_vocab.get(k) is not None:
                incorrect.append(k)
        for k in incorrect:
            self.word_vocab.pop(k)
        i = 0
        for k,v in self.word_vocab.items():
            self.word_vocab[k] = i
            i=i+1
        j = 0
        for k,v in self.bpe_vocab.items():
            self.bpe_vocab[k] = i + j
            j = j+1
        self.vocab_size = len(self.word_vocab)+len(self.bpe_vocab)
        print('after kick!')
        print(len(self.word_vocab))
        if self.custom_tokenizer and not dont_warn:
            print("WARNING! You've specified a non-default tokenizer.  You'll need to reassign it when you load the "
                  "model!")
        return {
            'byte_pairs': self.bpe_vocab,
            'words': self.word_vocab,
            'kwargs': {
                'vocab_size': self.vocab_size,
                'pct_bpe': self.pct_bpe,
                'silent': self._progress_bar is iter,
                'ngram_min': self.ngram_min,
                'ngram_max': self.ngram_max,
                'required_tokens': self.required_tokens,
                'strict': self.strict,
                'EOW': self.EOW,
                'SOW': self.SOW,
                'PAD': self.PAD,
                'UNK': self.UNK,
                # 'PAD': self.PAD,
            }
        }

    def savekikavocab(self,outpath):
        with open(outpath, 'w') as outfile:
            print(len(self.word_vocab))
            print(len(self.bpe_vocab))
            self.word_vocab.update(self.bpe_vocab)
            print(len(self.word_vocab))
            for k,v in enumerate(self.word_vocab):
                line = str(v)+'##'+str(k)+'\n'
                outfile.write(line)

    def save(self, outpath, dont_warn=False):
        # type: (Encoder, str, bool) -> None
        """ Serializes and saves encoder to provided path """
        with open(outpath, 'w') as outfile:
            json.dump(self.vocabs_to_dict(dont_warn), outfile)

    @classmethod
    def from_dict(cls, vocabs,letter_path):
        # type: (Any, Dict[str, Dict[str, int]]) -> Encoder
        """ Load encoder from dict produced with vocabs_to_dict """
        encoder = Encoder(**vocabs['kwargs'])
        encoder.word_vocab = vocabs['words']
        encoder.bpe_vocab = vocabs['byte_pairs']
        encoder.bpe_vocab[encoder.SOW] = 2
        encoder.bpe_vocab[encoder.EOW] = 3
        
        encoder.word_vocab.pop(encoder.SOW)
        encoder.word_vocab.pop(encoder.EOW)
        encoder.inverse_bpe_vocab = {v: k for k, v in encoder.bpe_vocab.items()}
        encoder.inverse_word_vocab = {v: k for k, v in encoder.word_vocab.items()}

        encoder.letters_dict = {}
        # print(self.letter_path)
        with open(letter_path,'r') as f:
            for line in f.readlines():
                l,n = line.split('##')
                encoder.letters_dict[l] = int(n)
        # print(encoder.letters_dict)
        for k,v in enumerate(encoder.letters_dict):
            encoder.letters.append(k)
        return encoder

    @classmethod
    def load(cls, in_path,letter_path):
        # type: (Any, str) -> Encoder
        """ Loads an encoder from path saved with save """
        with open(in_path) as infile:
            obj = json.load(infile)
        return cls.from_dict(obj,letter_path)
