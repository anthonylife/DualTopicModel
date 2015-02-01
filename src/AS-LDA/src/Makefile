CC = g++ -std=c++0x
CFLAGS = -Wall -O3 -fopenmp -Wcpp 

all: main.cpp lda.o ml_data.o text_tools.o
	$(CC) $(CFLAGS) -o ../as-lda main.cpp lda.o ml_data.o text_tools.o

lda.o: lda.cpp lda.h
	$(CC) $(CFLAGS) -c lda.cpp lda.h

ml_data.o: ml_data.cpp ml_data.h
	$(CC) $(CFLAGS) -c ml_data.cpp ml_data.h

text_tools.o: text_tools.cpp text_tools.h
	$(CC) $(CFLAGS) -c text_tools.cpp text_tools.h

clean:
	rm main
