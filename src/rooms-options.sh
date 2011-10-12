ITERS=10
ENSEMBLES=10
EPOCHS=40000

DD="rooms-options"
OD="options-rooms"
tmp_prefix="rc1"

# Make the directory
if [ ! -e $DD ]; then mkdir $DD; fi;

n=200
for o in $OD/*.options; do
    scheme="load"
    echo "Running options from $o..."
    PYTHONOPTIMIZE=3 python2 ./main.py $ITERS $ENSEMBLES $EPOCHS "MacroQ" "RoomsOptions:../domains/rooms1.txt:$scheme:$n:$o" $tmp_prefix
    mv "$tmp_prefix-return.dat" $DD/$(basename $o .options).return
done;
