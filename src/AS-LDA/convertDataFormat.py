#!/usr/bin/env python
#encoding=utf8

#Copyright [2014] [Wei Zhang]

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

###################################################################
# Date: 2014/7/2                                                  #
# Converting our raw specified data format to satisfy the input   #
#   requirements of LDA model.                                    #
###################################################################

import sys, csv, json, argparse, random
sys.path.append("../")
from collections import defaultdict

settings = json.loads(open("../../SETTINGS.json").read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='data_num', help='choose which data set to use')

    if len(sys.argv) != 3:
        print 'Command e.g.: python convertDataFormat.py -d 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.data_num == 0:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE1"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE1"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE1"]
        tr_review_path = "./data/tr_review_file1.dat"
        va_review_path = "./data/va_review_file1.dat"
        te_review_path = "./data/te_review_file1.dat"
        vocab_path = "./data/vocab1.dat"
    elif para.data_num == 1:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE2"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE2"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE2"]
        tr_review_path = "./data/tr_review_file2.dat"
        va_review_path = "./data/va_review_file2.dat"
        te_review_path = "./data/te_review_file2.dat"
        vocab_path = "./data/vocab2.dat"
    elif para.data_num == 2:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE3"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE3"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE3"]
        tr_review_path = "./data/tr_review_file3.dat"
        va_review_path = "./data/va_review_file3.dat"
        te_review_path = "./data/te_review_file3.dat"
        vocab_path = "./data/vocab3.dat"
    elif para.data_num == 3:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE4"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE4"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE4"]
        tr_review_path = "./data/tr_review_file4.dat"
        va_review_path = "./data/va_review_file4.dat"
        te_review_path = "./data/te_review_file4.dat"
        vocab_path = "./data/vocab4.dat"
    elif para.data_num == 4:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE5"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE5"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE5"]
        tr_review_path = "./data/tr_review_file5.dat"
        va_review_path = "./data/va_review_file5.dat"
        te_review_path = "./data/te_review_file5.dat"
        vocab_path = "./data/vocab5.dat"
    elif para.data_num == 5:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE6"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE6"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE6"]
        tr_review_path = "./data/tr_review_file6.dat"
        va_review_path = "./data/va_review_file6.dat"
        te_review_path = "./data/te_review_file6.dat"
        vocab_path = "./data/vocab6.dat"
    elif para.data_num == 7:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE8"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE8"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE8"]
        tr_review_path = "./data/tr_review_file7.dat"
        va_review_path = "./data/va_review_file7.dat"
        te_review_path = "./data/te_review_file7.dat"
        vocab_path = "./data/vocab7.dat"
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    words_cnt = defaultdict(int)
    for line in open(tr_data_path):
        parts = line.strip("\r\t\n").split(" ")
        for word in parts[4:]:
            if len(word) > 0:
                words_cnt[word] += 1

    saved_words = set([pair[0] for pair in sorted(words_cnt.items(), key=lambda x:x[1], reverse=True)][:settings["MAX_WORDS"]])
    word_dict = {}
    rword_dict = {}
    output_vocabulary = []
    for word in saved_words:
        wid = len(word_dict)
        word_dict[word] = wid
        rword_dict[wid] = word
        output_vocabulary.append(word)

    doc_num = 0
    wfd = open(tr_review_path, "w")
    for line in open(tr_data_path):
        parts =line.strip("\r\t\n").split(" ")
        if len(parts[4:]) == 0:
            continue
        valid_text = []
        for word in parts[5:]:
            if word in word_dict:
                valid_text.append(str(word_dict[word]))
        if len(valid_text) > 0:
            wfd.write("%s\n" % ("\t".join(valid_text)))
            doc_num += 1
    wfd.close()
    print doc_num

    doc_num = 0
    wfd = open(te_review_path, "w")
    for line in open(te_data_path):
        parts = line.strip("\r\t\n").split(" ")
        if len(parts[4:]) == 0:
            continue
        valid_text = []
        for word in parts[5:]:
            if word in word_dict:
                valid_text.append(str(word_dict[word]))
        if len(valid_text) > 0:
            wfd.write("%s\n" % ("\t".join(valid_text)))
            doc_num += 1
    wfd.close()
    print doc_num

    wfd = open(vocab_path, "w")
    for word in output_vocabulary:
        wfd.write("%s\n" % word)
    wfd.close()

if __name__ == "__main__":
    main()

