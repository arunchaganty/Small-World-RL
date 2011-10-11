ITERS=2
ENSEMBLES=2
EPOCHS=400

DD="rooms-scale"
scale="huge1"
tmp_prefix="rc-$scale"

# Make the directory
if [ ! -e $DD ]; then mkdir $DD; fi;

# Run without options
scheme=none
echo "Running $scheme..."
PYTHONOPTIMIZE=3 python2 ./main.py $ITERS $ENSEMBLES $EPOCHS "MacroQ" "RoomsOptions:../domains/rooms-scale/$scale.tsv:$scheme" $tmp_prefix
mv "$tmp_prefix-return.dat" $DD/$scale-$scheme.return
mv "$tmp_prefix-decisions.dat" $DD/$scale-$scheme.decisions

N="30% 50% 70 90%"
for n in $N; do
  for scheme in "betweenness" "random-path"; do
      echo "Running $scheme with $n options..."
      PYTHONOPTIMIZE=3 python2 ./main.py $ITERS $ENSEMBLES $EPOCHS "MacroQ" "RoomsOptions:../domains/rooms-scale/$scale.tsv:$scheme:$n" $tmp_prefix
      mv "$tmp_prefix-return.dat" $DD/$scale-$scheme-$n.return
      mv "$tmp_prefix-decisions.dat" $DD/$scale-$scheme-$n.decisions
  done;

  scheme="small-world"
  # Run for a bunch of 'r'
  for r in 0.75 1.0 2.0; do
      echo "Running $scheme(r=$r) with $n options..."
      PYTHONOPTIMIZE=3 python2 ./main.py $ITERS $ENSEMBLES $EPOCHS "MacroQ" "RoomsOptions:../domains/rooms-scale/$scale.tsv:$scheme:$n:$r" $tmp_prefix
      mv "$tmp_prefix-return.dat" $DD/$scale-$scheme-$n-$r.return
      mv "$tmp_prefix-decisions.dat" $DD/$scale-$scheme-$n-$r.decisions
  done;
done;
