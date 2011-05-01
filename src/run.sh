# Run 

AGENT=MacroQ
TRIALS=100
EPISODES=2000
GAMMA=0.9 #0.99
RATE=0.1 #0.01
SCHEMES="small-world betweenness manual none random optimal" #none
DOMAIN="taxi2.txt taxi1.txt"

alpha=0.8
e=0.1

for gamma in $GAMMA; do
    for rate in $RATE; do
        for scheme in $SCHEMES; do
            for domain in $DOMAIN; do
                echo "# Running with g,r,s,d = $gamma, $rate, $scheme, $domain"
                cmd="python2 ./main.py $EPISODES "$AGENT:$gamma:$alpha:$e:$rate" "TaxiOptions:./data/$domain:$scheme" "
                out="./output/$(basename $domain .txt)-$scheme-$rate-$gamma"
                if [ ! -e $out ]; then mkdir $out; fi;
                for i in `seq 1 $TRIALS`; do
                    echo $(calc "round($i/$TRIALS * 100)")
                    $cmd > $out/$i.dat;
                done;
            done;
        done;
    done;
done | zenity --progress;

