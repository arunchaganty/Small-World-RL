PYTHON=python2
ITERS=3
EPOCHS="1e4" #"1e5 1e6 1e8"
N=100

DD="rooms-options"
tmp_prefix="tmp1"

# Make the directory
if [ ! -e $DD ]; then mkdir $DD; fi;

for e in $EPOCHS; do 
    scheme="small-world"
    PYTHONOPTIMIZE=3 $PYTHON ./make_options.py $e $N "$scheme:0.75" "MacroQ" "Rooms:../domains/rooms1.txt" $tmp_prefix
    mv "$tmp_prefix.options" "$DD/$tmp_prefix-$e.options"
done;
