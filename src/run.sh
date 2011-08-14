# Run 

AGENT=IntraOptionQ
TRIALS=500
EPISODES=2000
GAMMA=0.99 #0.99
RATE=0.01 #0.01
SCHEMES="small-world betweenness none random ozgur-random optimal" #none
DOMAIN="taxi1.txt"
OCount=20

alpha=0.8
e=0.1

SD=./scripts

for gamma in $GAMMA; do
    for rate in $RATE; do
        for scheme in $SCHEMES; do
            for domain in $DOMAIN; do
                echo "# Running with g,r,s,d = $gamma, $rate, $scheme, $domain"
                cmd="python2 ./main.py $EPISODES "$AGENT:$gamma:$alpha:$e:$rate" "TaxiOptions:./data/$domain:$scheme:$OCount" "
                out="output-$OCount/${AGENT}-${gamma}/$(basename $domain .txt)-$scheme-$rate-$gamma"
                if [ ! -e $out ]; then mkdir $out; fi;
                for i in `seq 1 $TRIALS`; do
                    echo $(calc "round($i/$TRIALS * 100)")
                    $cmd > $out/$i.dat;
                    $SD/add_x -w $out/$i.dat;
                done;
            done;
        done;
    done;
done | zenity --progress;

