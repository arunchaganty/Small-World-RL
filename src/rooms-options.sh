PYTHON=python2
ITERS=3
EPOCHS="1e5 1e6 1e8"
N=30

DD="rooms-options"
tmp_prefix="tmp1"

# Make the directory
if [ ! -e $DD ]; then mkdir $DD; fi;

for e in $EPOCHS; do 
    scheme="betweenness"
    echo PYTHONOPTIMIZE=3 $PYTHON ./make_options.py $e $N "$scheme" "MacroQ" "Rooms:../domains/rooms1.txt" $tmp_prefix
    mv "$tmp_prefix.options" "$DD/$tmp_prefix-$e.options"
done;
