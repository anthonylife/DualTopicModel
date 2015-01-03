#/bin/sh

make train_target=train
if [ $? -ne 0 ]; then
    exit 1
fi

echo "main"
#./main -d 1 -r True 
./main -d 4 -r True

#valgrind --tool=memcheck --leak-check=full --show-reachable=yes --error-limit=no --log-file=valgrind.log ./demo.sh
