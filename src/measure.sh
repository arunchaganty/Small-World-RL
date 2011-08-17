commitish=`git log -n1 | head -n1`;

/usr/bin/time -f "%e %M %I" ./measure_.sh 2> .a
stats=`cat .a`;
rm .a;

echo $commitish $stats
