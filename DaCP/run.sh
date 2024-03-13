make
rm -rf result.csv
for i in ../../matrix/misc/*.mtx
do
    ./YYSpTRSV $i
done