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
# Date: 2015/1/8                                                  #
# Applying diverse methods for classification                     #
###################################################################

import csv, json, sys, argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, average_precision_score, classification_report

settings = json.loads(open("../../SETTINGS.json").read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, action='store',
            dest='target', help='for validation or test dataset')
    parser.add_argument('-m', type=int, action='store',
            dest='model', help='choosing which model to use')

    if len(sys.argv) != 5:
        print 'Command e.g.: python train.py -d 0(1,...) -m 0(1)'
        sys.exit(1)

    para = parser.parse_args()
    if para.target == 0:
        if para.model == 0:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE1-DALDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE1-DALDA"]
        elif para.model == 1:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE1-LDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE1-LDA"]
        else:
            print 'Invalid train data target choice...'
            sys.exit(1)
    elif para.target == 1:
        if para.model == 0:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE2-DALDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE2-DALDA"]
        elif para.model == 1:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE2-LDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE2-LDA"]
        else:
            print 'Invalid train data target choice...'
            sys.exit(1)
    elif para.target == 2:
        if para.model == 0:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE3-DALDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE3-DALDA"]
        elif para.model == 1:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE3-LDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE3-LDA"]
        else:
            print 'Invalid train data target choice...'
            sys.exit(1)
    elif para.target == 3:
        if para.model == 0:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE4-DALDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE4-DALDA"]
        elif para.model == 1:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE4-LDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE4-LDA"]
        else:
            print 'Invalid train data target choice...'
            sys.exit(1)
    elif para.target == 4:
        if para.model == 0:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE5-DALDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE5-DALDA"]
        elif para.model == 1:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE5-LDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE5-LDA"]
        else:
            print 'Invalid train data target choice...'
            sys.exit(1)
    elif para.target == 5:
        if para.model == 0:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE6-DALDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE6-DALDA"]
        elif para.model == 1:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE6-LDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE6-LDA"]
        else:
            print 'Invalid train data target choice...'
            sys.exit(1)
    elif para.target == 6:
        if para.model == 0:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE7-DALDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE7-DALDA"]
        elif para.model == 1:
            in_tr_path = settings["ROOT_PATH"] + settings["TRAIN_CL_FILE7-LDA"]
            in_te_path = settings["ROOT_PATH"] + settings["TEST_CL_FILE7-LDA"]
        else:
            print 'Invalid train data target choice...'
            sys.exit(1)
    else:
        print 'Invalid train data target choice...'
        sys.exit(1)
    tr_features = [map(float, entry.strip("\r\t\n").split(" ")[3:-1]) for entry in open(in_tr_path)]
    tr_labels = [int(entry.strip("\r\t\n").split(" ")[2]) for entry in open(in_tr_path)]
    te_features = [map(float, entry.strip("\r\t\n").split(" ")[3:-1]) for entry in open(in_te_path)]
    te_labels = [int(entry.strip("\r\t\n").split(" ")[2]) for entry in open(in_te_path)]

    '''classifier = GradientBoostingClassifier(n_estimators=50,
                                        verbose=2,
                                        min_samples_split=10,
                                        random_state=1)'''
    '''classifier = RandomForestClassifier(n_estimators=100,
                                        verbose=2,
                                        n_jobs=1,
                                        min_samples_split=10,
                                        random_state=1)'''
    '''classifier = LogisticRegression(penalty='l2',
                                    dual=False,
                                    tol=0.0001,
                                    fit_intercept=True,
                                    class_weight=None,
                                    random_state=None)'''
    classifier = LinearSVC(penalty='l2',
                           loss='l2',
                           C=0.1)
    classifier.fit(tr_features, tr_labels)
    pred_labels = classifier.predict(te_features)

    accu = accuracy_score(te_labels, pred_labels)
    aps = average_precision_score(te_labels, pred_labels)
    print "Accuracy of classification: %f, average precision score: %f" % (accu, aps)
    print(classification_report(te_labels, pred_labels))

if __name__ == "__main__":
    main()
