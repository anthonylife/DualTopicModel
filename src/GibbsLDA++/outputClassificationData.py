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
# Date: 2015/1/7                                                  #
# Output classification data.                                     #
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
        print 'Command e.g.: python OutputClassificationData -d 0(1)'
        sys.exit(1)

    train_topic_path = "./model-final.theta"
    test_topic_path = "./test-final.theta"
    para = parser.parse_args()
    if para.data_num == 0:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE1"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE1"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE1"]
        out_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE1-LDA"]
        out_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE1-LDA"]
    elif para.data_num == 1:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE2"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE2"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE2"]
        out_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE2-LDA"]
        out_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE2-LDA"]
    elif para.data_num == 2:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE3"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE3"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE3"]
        out_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE3-LDA"]
        out_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE3-LDA"]
    elif para.data_num == 3:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE4"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE4"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE4"]
        out_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE4-LDA"]
        out_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE4-LDA"]
    elif para.data_num == 4:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE5"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE5"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE5"]
        out_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE5-LDA"]
        out_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE5-LDA"]
    elif para.data_num == 5:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE6"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE6"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE6"]
        out_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE6-LDA"]
        out_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE6-LDA"]
    elif para.data_num == 6:
        tr_data_path = settings["ROOT_PATH"]+settings["TRAIN_DATA_FILE7"]
        va_data_path = settings["ROOT_PATH"]+settings["VALI_DATA_FILE7"]
        te_data_path = settings["ROOT_PATH"]+settings["TEST_DATA_FILE7"]
        out_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE7-LDA"]
        out_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE7-LDA"]
    else:
        print 'Invalid choice of dataset'
        sys.exit(1)

    outputline = []
    for line in open(tr_data_path):
        outputline.append(" ".join(line.strip("\r\t\n").split(" ")[:3]))

    wfd = open(out_tr_path, "w")
    for line1, line2 in zip(outputline, open(train_topic_path)):
        wfd.write("%s" % (line1+" "+line2))

    outputline = []
    for line in open(te_data_path):
        outputline.append(" ".join(line.strip("\r\t\n").split(" ")[:3]))

    wfd = open(out_te_path, "w")
    for line1, line2 in zip(outputline, open(test_topic_path)):
        wfd.write("%s" % (line1+" "+line2))

if __name__ == "__main__":
    main()

