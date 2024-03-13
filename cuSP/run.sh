rm -rf result.csv
for i in ../../matrix/misc/*.mtx
do
    ./main $i
done