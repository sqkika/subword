import sys
import codecs
from encoder import Encoder
import numpy as np



language = 'cs'
train_data_path = '/home/pubsrv/data/train_data/cs/train_data_cs_user_web_mapletters_30per/train_data'
dev_data_path = '/home/pubsrv/data/train_data/cs/train_data_cs_user_web_mapletters_30per/dev_data'
base_outdir = '/home/sunquan/data/bpedata/cs/'



pctbpe = 0.2
letter_path = base_outdir+'vocab_in_letters'
json_vocab_path =base_outdir+('%1f_vocab.bpe')%pctbpe
lstmvocab = base_outdir+'vocab_in_words'

train_lm_ids_data = base_outdir+'train_in_ids_lm'
train_letter_ids_data = base_outdir+'train_in_ids_letters'

dev_lm_ids_data = base_outdir+'dev_in_ids_lm'
dev_letter_ids_data = base_outdir+'dev_in_ids_letters'


#mode 1 : generator letter vocab, and then u can modify the lettervocab by hands
#mode 2 : generator word vocab according to letter vocab
#mode 3 : generator kika lstm dataset according to the letter vocab and word vocab, and then ,u can train your kikaengine model on the data
mode = 3

def get_wordlist(data_path):
	lm_word_list = []
	count = 0
	wordcounts = 0
	with codecs.open(data_path, "r", encoding="utf-8") as f:
	    for line in f.readlines():
	        # print(line)
	        _,lm_sentense = line.strip().split("|#|")
	        lm_words = lm_sentense.split("\t")
	        # print(lm_words)
	        wordcounts+=len(lm_words)
	        # if len(lm_words) <30:
	        #     continue
	        count +=1
	        # print(count)
	        lm_word_list.append(lm_sentense)
	return lm_word_list,wordcounts


def count_unk(data_path,lm_ids_data,letter_ids_data,encoder):
	count = 0
	lenth = 0
	usrlenth = 0
	lm = codecs.open(lm_ids_data, "w", encoding="utf-8")
	letterfp = codecs.open(letter_ids_data, "w", encoding="utf-8")
	with codecs.open(data_path, "r", encoding="utf-8") as f:
	    for line in f.readlines():
	        # print(line)
	        usr_sentense,lm_sentense = line.strip().split("|#|")
	        # id_list = np.array(next(encoder.transform([lm_sentense])))
	        id_list = encoder.transform([lm_sentense])
	        lm_line_in = '1'
	        lm_line_out = '0'
	        for ids in id_list:
	        	if ids==2 or ids ==3:
	        		continue
	        	lm_line_in+=' '
	        	lm_line_in+=str(ids)
	        	lm_line_out+=' '
	        	lm_line_out+=str(ids)
	        lm_line=lm_line_in+'#'+lm_line_out+'\n'

	        lm.write(lm_line)
	        token_lm_line = encoder.tokenize(lm_sentense)
	        token_letters_line = encoder.transformletters(token_lm_line)

	       	# print(lm_sentense)
	        # print(token_lm_line)
	        # print(id_list)
	        # print(lm_line)
	        # print(token_letters_line)
	        
	        letterfp.write(token_letters_line)
	        # usr_id_list = np.array(next(encoder.transform([usr_sentense])))
	        lenth+=len(id_list)

	        
	        # print(encoder.tokenize(lm_sentense))
	        # print(id_list)
	        print(count)
	        for wid in id_list:
	        	if wid<2:
	        		count+=1
	return count,lenth

lm_word_list,wordcounts = get_wordlist(train_data_path)

if mode==1:#generator letters
	encoder = Encoder(20000, pct_bpe=pctbpe,letter_path = letter_path)  # params chosen for demonstration purposes
	encoder.create_lettervocab(lm_word_list)
elif mode==2:#generator bpe and word vocab
	encoder = Encoder(20000, pct_bpe=pctbpe, ngram_min=2, ngram_max=10, letter_path = letter_path)  # params chosen for demonstration purposes
	encoder.fit(lm_word_list)
	encoder.save(json_vocab_path, dont_warn=False)
	encoder.savekikavocab(lstmvocab)
elif mode ==3:#generator ids_data
	encoder = Encoder.load(json_vocab_path,letter_path)
	count,lenth = count_unk(train_data_path,train_lm_ids_data,train_letter_ids_data,encoder)
	count_unk(dev_data_path,dev_lm_ids_data,dev_letter_ids_data,encoder)
	print(count)
	print(lenth)




# # example = "Vizzini: He didn't fall? INCONCEIVABLE!"
# example = 'Llevo	desde	las	6	ek	el	estacionamiento	del	metro	mall	y	estaba	tan	aburrida	que	mejor	me	puse	a	cantar	.	¿A	qué	va	todo	esto	?	Que	yo	estaba	cantando	a	todo	pulmón	y	no	me	daba	cuenta	que	las	pláticas	de	la	gente	se	escuchan	fácilmente	,	y	eso	significa	que	fijo	me	escucharon	cantar	.	Encima	lo	estaba	haciendo	mal	,	y	de	paso	,	esta	hora	que	llevo	en	el	estacionamiento	,	no	me	di	cuenta	que	había	wifi	sino	hasta	hace	unos	minutos	.	_	.'
# print(encoder.tokenize(example))
# # ['__sow', 'vi', 'z', 'zi', 'ni', '__eow', '__sow', ':', '__eow', 'he', 'didn', "'", 't', 'fall', '__sow', '?', '__eow', '__sow', 'in', 'co', 'n', 'ce', 'iv', 'ab', 'le', '__eow', '__sow', '!', '__eow']
# print(next(encoder.transform([example])))
# # [24, 108, 82, 83, 71, 25, 24, 154, 25, 14, 10, 11, 12, 13, 24, 85, 25, 24, 140, 59, 39, 157, 87, 165, 114, 25, 24, 148, 25]
# print(next(encoder.inverse_transform(encoder.transform([example]))))
# # vizzini : he didn ' t fall ? inconceivable !