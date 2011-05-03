# Run 

AGENT=MacroQ
TRIALS=200
EPISODES=1500
GAMMA=0.99 #0.99
RATE=0.01 #0.01
SCHEMES="manual optimal" #none
DOMAIN="taxi1.txt"

alpha=0.8
e=0.1

for gamma in $GAMMA; do
    for rate in $RATE; do
        for scheme in $SCHEMES; do
            for domain in $DOMAIN; do
                echo "# Running with g,r,s,d = $gamma, $rate, $scheme, $domain"
                cmd="python2 ./main.py $EPISODES "$AGENT:$gamma:$alpha:$e:$rate" "TaxiOptions:./data/$domain:$scheme" "
                out="./${AGENT}-${gamma}/$(basename $domain .txt)-$scheme-$rate-$gamma"
                if [ ! -e $out ]; then mkdir $out; fi;
                for i in `seq 1 $TRIALS`; do
                    echo $(calc "round($i/$TRIALS * 100)")
                    $cmd > $out/$i.dat;
                done;
            done;
        done;
    done;
done | zenity --progress;

