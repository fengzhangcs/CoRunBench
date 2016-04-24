program_dir="/home/pacman/CoRunBench/"
result_dir="/home/pacman/CoRunBench/result/"

echo "program dir = $program_dir"
echo "result dir = $result_dir"

cd $program_dir/leukocyte/OpenCL
bash run_dwarves.sh | tee $result_dir/1leukocyte.txt

cd $program_dir/heartwall
bash run_dwarves.sh | tee $result_dir/2heartwall.txt

cd $program_dir/cfd
bash run_dwarves.sh | tee $result_dir/3cfd.txt

cd $program_dir/lud/ocl
bash run_dwarves.sh | tee $result_dir/4lud.txt

cd $program_dir/hotspot
bash run_dwarves.sh | tee $result_dir/5hotspot.txt

cd $program_dir/backprop
bash run_dwarves.sh | tee $result_dir/6backprop.txt

cd $program_dir/nw
bash run_dwarves.sh | tee $result_dir/7nw.txt

cd $program_dir/kmeans
bash run_dwarves.sh | tee $result_dir/8kmeans.txt

cd $program_dir/bfs
bash run_dwarves.sh | tee $result_dir/9bfs.txt

cd $program_dir/srad
bash run_dwarves.sh | tee $result_dir/10srad.txt

cd $program_dir/streamcluster
bash run_dwarves.sh | tee $result_dir/11streamcluster.txt

cd $program_dir/particlefilter
bash run_dwarves.sh | tee $result_dir/12particlefilter.txt

cd $program_dir/pathfinder
bash run_dwarves.sh | tee $result_dir/13pathfinder.txt

cd $program_dir/gaussian
bash run_dwarves.sh | tee $result_dir/14gaussian.txt

cd $program_dir/nn
bash run_dwarves.sh | tee $result_dir/15nn.txt

cd $program_dir/lavaMD
bash run_dwarves.sh | tee $result_dir/16lavamd.txt

cd $program_dir/myocyte
bash run_dwarves.sh | tee $result_dir/17myocyte.txt

cd $program_dir/b+tree
bash run_dwarves.sh | tee $result_dir/18btree.txt

cd $program_dir/dwt2d
bash run_dwarves.sh | tee $result_dir/19dwt2d.txt

cd $program_dir/hybridsort
bash run_dwarves.sh | tee $result_dir/20hybridsort.txt

cd $result_dir
mkdir -p tempt
grep CAUT 1[a-z]*.txt | awk '{print $3}' > tempt/0.txt
grep CAUT 1[a-z]*.txt | awk '{print $5}' > tempt/1.txt
grep CAUT 2[a-z]*.txt | awk '{print $5}' > tempt/2.txt
grep CAUT 3[a-z]*.txt | awk '{print $5}' > tempt/3.txt
grep CAUT 4[a-z]*.txt | awk '{print $5}' > tempt/4.txt
grep CAUT 5[a-z]*.txt | awk '{print $5}' > tempt/5.txt
grep CAUT 6[a-z]*.txt | awk '{print $5}' > tempt/6.txt
grep CAUT 7[a-z]*.txt | awk '{print $5}' > tempt/7.txt
grep CAUT 8[a-z]*.txt | awk '{print $5}' > tempt/8.txt
grep CAUT 9[a-z]*.txt | awk '{print $5}' > tempt/9.txt
grep CAUT 10[a-z]*.txt | awk '{print $5}' > tempt/10.txt
grep CAUT 11[a-z]*.txt | awk '{print $5}' > tempt/11.txt
grep CAUT 12[a-z]*.txt | awk '{print $5}' > tempt/12.txt
grep CAUT 13[a-z]*.txt | awk '{print $5}' > tempt/13.txt
grep CAUT 14[a-z]*.txt | awk '{print $5}' > tempt/14.txt
grep CAUT 15[a-z]*.txt | awk '{print $5}' > tempt/15.txt
grep CAUT 16[a-z]*.txt | awk '{print $5}' > tempt/16.txt
grep CAUT 17[a-z]*.txt | awk '{print $5}' > tempt/17.txt
grep CAUT 18[a-z]*.txt | awk '{print $5}' > tempt/18.txt
grep CAUT 19[a-z]*.txt | awk '{print $5}' > tempt/19.txt
grep CAUT 20[a-z]*.txt | awk '{print $5}' > tempt/20.txt




echo "%, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20" > res.csv
echo "%, leukocyte, heartwall, cfd, lud, hotspot, backpropagation, nw, kmeans, bfs, srad, streamcluster, particlefilter, pathfinder, gaussian, nn, lavamd, myocyte, b+tree, gpudwt, hybridsort" >> res.csv
paste -d ,  tempt/0.txt tempt/1.txt tempt/2.txt tempt/3.txt tempt/4.txt tempt/5.txt tempt/6.txt tempt/7.txt tempt/8.txt tempt/9.txt tempt/10.txt tempt/11.txt tempt/12.txt tempt/13.txt tempt/14.txt tempt/15.txt tempt/16.txt tempt/17.txt tempt/18.txt tempt/19.txt tempt/20.txt >> res.csv

echo "Finished! Please look at the res.csv using EXCEL."
echo "The first column is the partitioning ratio. The time is in ms."
