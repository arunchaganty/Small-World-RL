# Run 

AGENT=MacroQ
TRIALS=200
EPISODES=10000
GAMMA=1 #0.99
RATE=0 #0.01
SCHEMES="small-world betweenness none ozgur-random random none"
DOMAIN="taxi1.txt"
OCount=20

alpha=0.1
e=0.1

SD=./scripts

for gamma in $GAMMA; do
    for rate in $RATE; do
        for scheme in $SCHEMES; do
            for domain in $DOMAIN; do
                echo "# Running with g,r,s,d = $gamma, $rate, $scheme, $domain"
                cmd="python ./main.py $EPISODES "$AGENT:$gamma:$alpha:$e:$rate" "TaxiOptions:./data/$domain:$scheme:$OCount" "
                out="output-$OCount/${AGENT}-${gamma}/$(basename $domain .txt)-$scheme-$rate-$gamma"
                if [ ! -e $out ]; then mkdir -p $out; fi;
                for i in `seq 1 $TRIALS`; do
                    echo $(echo "$i.0/$TRIALS.0 * 100" | bc -l)
                    $cmd > $out/$i.dat;
                    mv rl.out $out/$i.rwd;
                    $SD/add_x -w $out/$i.dat;
                done;
            done;
        done;
    done;
done;

