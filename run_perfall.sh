program_dir="/home/pacman/CoRunBench/"
result_dir="/home/pacman/CoRunBench/perfresult/"

echo "program dir = $program_dir"
echo "result dir = $result_dir"
i=1
echo $i; let i="$i + 1";
cd $program_dir/leukocyte/OpenCL
bash run_perf.sh > $result_dir/1leukocyte.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/heartwall
bash run_perf.sh > $result_dir/2heartwall.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/cfd
bash run_perf.sh > $result_dir/3cfd.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/lud/ocl
bash run_perf.sh > $result_dir/4lud.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/hotspot
bash run_perf.sh > $result_dir/5hotspot.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/backprop
bash run_perf.sh > $result_dir/6backprop.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/nw
bash run_perf.sh > $result_dir/7nw.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/kmeans
bash run_perf.sh > $result_dir/8kmeans.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/bfs
bash run_perf.sh > $result_dir/9bfs.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/srad
bash run_perf.sh > $result_dir/10srad.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/streamcluster
bash run_perf.sh > $result_dir/11streamcluster.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/particlefilter
bash run_perf.sh > $result_dir/12particlefilter.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/pathfinder
bash run_perf.sh > $result_dir/13pathfinder.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/gaussian
bash run_perf.sh > $result_dir/14gaussian.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/nn
bash run_perf.sh > $result_dir/15nn.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/lavaMD
bash run_perf.sh > $result_dir/16lavamd.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/myocyte
bash run_perf.sh > $result_dir/17myocyte.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/b+tree
bash run_perf.sh > $result_dir/18btree.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/dwt2d
bash run_perf.sh > $result_dir/19dwt2d.txt 2>&1

echo $i; let i="$i + 1";
cd $program_dir/hybridsort
bash run_perf.sh > $result_dir/20hybridsort.txt 2>&1


cd $result_dir

mkdir -p tempt
word="L1-dcache-load-misses"
i=4

grep "$word" 1[a-z]*.txt | awk '{print $3}' > tempt/0.txt
grep "$word" 1[a-z]*.txt | awk '{print $'$i'}' > tempt/1.txt
grep "$word" 2[a-z]*.txt | awk '{print $'$i'}' > tempt/2.txt
grep "$word" 3[a-z]*.txt | awk '{print $'$i'}' > tempt/3.txt
grep "$word" 4[a-z]*.txt | awk '{print $'$i'}' > tempt/4.txt
grep "$word" 5[a-z]*.txt | awk '{print $'$i'}' > tempt/5.txt
grep "$word" 6[a-z]*.txt | awk '{print $'$i'}' > tempt/6.txt
grep "$word" 7[a-z]*.txt | awk '{print $'$i'}' > tempt/7.txt
grep "$word" 8[a-z]*.txt | awk '{print $'$i'}' > tempt/8.txt
grep "$word" 9[a-z]*.txt | awk '{print $'$i'}' > tempt/9.txt
grep "$word" 10[a-z]*.txt | awk '{print $'$i'}' > tempt/10.txt
grep "$word" 11[a-z]*.txt | awk '{print $'$i'}' > tempt/11.txt
grep "$word" 12[a-z]*.txt | awk '{print $'$i'}' > tempt/12.txt
grep "$word" 13[a-z]*.txt | awk '{print $'$i'}' > tempt/13.txt
grep "$word" 14[a-z]*.txt | awk '{print $'$i'}' > tempt/14.txt
grep "$word" 15[a-z]*.txt | awk '{print $'$i'}' > tempt/15.txt
grep "$word" 16[a-z]*.txt | awk '{print $'$i'}' > tempt/16.txt
grep "$word" 17[a-z]*.txt | awk '{print $'$i'}' > tempt/17.txt
grep "$word" 18[a-z]*.txt | awk '{print $'$i'}' > tempt/18.txt
grep "$word" 19[a-z]*.txt | awk '{print $'$i'}' > tempt/19.txt
grep "$word" 20[a-z]*.txt | awk '{print $'$i'}' > tempt/20.txt




echo "%, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20"
echo "%, leukocyte, heartwall, cfd, lud, hotspot, backpropagation, nw, kmeans, bfs, srad, streamcluster, particlefilter, pathfinder, gaussian, nn, lavamd, myocyte, b+tree, gpudwt, hybridsort"
paste -d ,  tempt/0.txt tempt/1.txt tempt/2.txt tempt/3.txt tempt/4.txt tempt/5.txt tempt/6.txt tempt/7.txt tempt/8.txt tempt/9.txt tempt/10.txt tempt/11.txt tempt/12.txt tempt/13.txt tempt/14.txt tempt/15.txt tempt/16.txt tempt/17.txt tempt/18.txt tempt/19.txt tempt/20.txt

rm -rf tempt
