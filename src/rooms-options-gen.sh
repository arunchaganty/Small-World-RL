PYTHON=python2
ITERS=3
EPOCHS="1e5 1e6 1e7"
r=0.75

DD="options-rooms"
tmp_prefix="tmp1"

# Make the directory
if [ ! -e $DD ]; then mkdir $DD; fi;

for n in 100; do
  for e in $EPOCHS; do 
    # Betweenness
    for scheme in "betweenness" "optimal-betweenness"; do
      echo "Building $n $scheme options..."
      PYTHONOPTIMIZE=3 $PYTHON ./make_options.py $e $n "$scheme" "MacroQ" "Rooms:../domains/rooms1.txt" $tmp_prefix
      mv "$tmp_prefix.options" "$DD/$scheme-$e.options"
    done;
    for scheme in "small-world" "optimal-small-world"; do
      echo "Building $n $scheme(r=$r) options..."
      PYTHONOPTIMIZE=3 $PYTHON ./make_options.py $e $n "$scheme:$r" "MacroQ" "Rooms:../domains/rooms1.txt" $tmp_prefix
      mv "$tmp_prefix.options" "$DD/$scheme-$e-$r.options"
    done;
  done;
done;
