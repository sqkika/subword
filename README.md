#mode 1 : generator letter vocab, and then u can modify the lettervocab by hands
#mode 2 : generator word vocab according to letter vocab
#mode 3 : generator kika lstm dataset according to the letter vocab and word vocab, and then ,u can train your kikaengine model on the data





#先使用运行mbpe.py ，mode1 生成字母表，mode2 生成单词表， mode3 生成数据；然后按照kika训练方式运行其他代码即可


#使用时，可以不用mode 1 生成letter词表，直接使用kika词表，注意在letter词表中加上'_'