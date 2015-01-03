CC = g++ -std=c++0x
CFLAGS = -Wall -O3 -fopenmp -Wcpp 

all: ${train_target}

train: main.cpp ../utils.hpp model.hpp
	$(CC) $(CFLAGS) -o main main.cpp 

clean:
	rm main
