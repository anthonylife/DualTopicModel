#!/bin/sh

echo "First, convert raw data format to satisfy the requirements of Gibbs LDA"
#python convertDataFormat.py -d 0
python convertDataFormat.py -d 1
#python convertDataFormat.py -d 2
#python convertDataFormat.py -d 3
#python convertDataFormat.py -d 5
#python convertDataFormat.py -d 6
#python convertDataFormat.py -d 7

if [ $? -ne 0 ]; then
    exit 1
fi

echo "Second, train by AS-LDA"
#echo "Yelp review"
#./as-lda -c data/tr_review_file1.dat -v data/vocab1.dat -e -m ./models -k 40 -m 1000
echo "Food review"
./as-lda -c data/tr_review_file2.dat -v data/vocab2.dat -e -m ./models -k 40 -m 1000
#echo "Movie review"
#./as-lda -c data/tr_review_file3.dat -v data/vocab3.dat -e -m ./models -k 40 -m 1000
#echo "Arts review"
#./as-lda -c data/tr_review_file4.dat -v data/vocab4.dat -e -m ./models -k 40 -m 1000
#echo "Yelp tip"
#./as-lda -c data/tr_review_file5.dat -v data/vocab5.dat -e -m ./models -k 40 -m 1000
#echo "Cellartracker review"
#./as-lda -c data/tr_review_file6.dat -v data/vocab6.dat -e -m ./models -k 40 -m 1000
#echo "Video review"
#./as-lda -c data/tr_review_file7.dat -v data/vocab7.dat -e -m ./models -k 40 -m 1000

if [ $? -ne 0 ]; then
    exit 1
fi

echo "Second, test by AS-LDA"
#echo "Yelp review"
#./as-lda -c data/te_review_file1.dat -v data/vocab1.dat -i -m ./models -k 40 -m 200
echo "Food review"
./as-lda -c data/te_review_file2.dat -v data/vocab2.dat -i -m ./models -k 40 -m 200
#echo "Movie review"
#./as-lda -c data/te_review_file3.dat -v data/vocab3.dat -i -m ./models -k 40 -m 200
#echo "Arts review"
#./as-lda -c data/te_review_file4.dat -v data/vocab4.dat -i -m ./models -k 40 -m 200
#echo "Yelp tip"
#./as-lda -c data/te_review_file5.dat -v data/vocab5.dat -i -m ./models -k 40 -m 200
#echo "Cellartracker review"
#./as-lda -c data/te_review_file6.dat -v data/vocab6.dat -i -m ./models -k 40 -m 200
#echo "Video review"
#./as-lda -c data/te_review_file7.dat -v data/vocab7.dat -i -m ./models -k 40 -m 200 

