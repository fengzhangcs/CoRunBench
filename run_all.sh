program_dir="/home/pacman/CoRunBench/"
result_dir="/home/pacman/CoRunBench/result/"
result_parboil_dir="$result_dir/parboil/"
mkdir -p $result_dir
mkdir -p $result_parboil_dir

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


cd $program_dir/2DCONV
bash run_dwarves.sh | tee $result_dir/21_2DCONV.txt

cd $program_dir/3DCONV
bash run_dwarves.sh | tee $result_dir/22_3DCONV.txt

cd $program_dir/ATAX
bash run_dwarves.sh | tee $result_dir/23_ATAX.txt

cd $program_dir/COVAR
bash run_dwarves.sh | tee $result_dir/24_COVAR.txt

cd $program_dir/GEMM
bash run_dwarves.sh | tee $result_dir/25_GEMM.txt

cd $program_dir/GRAMSCHM
bash run_dwarves.sh | tee $result_dir/26_GRAMSCHM.txt

cd $program_dir/SYR2K
bash run_dwarves.sh | tee $result_dir/27_SYR2K.txt

cd $program_dir/2MM
bash run_dwarves.sh | tee $result_dir/28_2MM.txt

cd $program_dir/3MM
bash run_dwarves.sh | tee $result_dir/29_3MM.txt

cd $program_dir/BICG
bash run_dwarves.sh | tee $result_dir/30_BICG.txt

cd $program_dir/CORR
bash run_dwarves.sh | tee $result_dir/31_CORR.txt

cd $program_dir/FDTD-2D
bash run_dwarves.sh | tee $result_dir/32_FDTD-2D.txt

cd $program_dir/GESUMMV
bash run_dwarves.sh | tee $result_dir/33_GESUMMV.txt

cd $program_dir/MVT
bash run_dwarves.sh | tee $result_dir/34_MVT.txt

cd $program_dir/SYRK
bash run_dwarves.sh | tee $result_dir/35_SYRK.txt






cd $program_dir/parboil/
bash  run_compileall.sh
for i in histo  lbm  mri-gridding  mri-q  sad  sgemm  spmv  
do
  b=`echo "$i.txt"`
  echo $i $b
  cd $program_dir/parboil/$i
  bash run_zf.sh | tee $result_parboil_dir/$b
done
cd $result_parboil_dir
mkdir -p tempt
for i in histo.txt  lbm.txt  mri-gridding.txt  mri-q.txt  sad.txt  sgemm.txt  spmv.txt  
do
  grep CAUT $i | awk '{print $5}' > tempt/$i
done






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

grep CAUT 21_*[A-Z].txt | awk '{print $7}' > tempt/21.txt
grep CAUT 22_*[A-Z].txt | awk '{print $7}' > tempt/22.txt
grep CAUT 23_*[A-Z].txt | awk '{print $7}' > tempt/23.txt
grep CAUT 24_*[A-Z].txt | awk '{print $7}' > tempt/24.txt
grep CAUT 25_*[A-Z].txt | awk '{print $7}' > tempt/25.txt
grep CAUT 26_*[A-Z].txt | awk '{print $7}' > tempt/26.txt
grep CAUT 27_*[A-Z].txt | awk '{print $7}' > tempt/27.txt
grep CAUT 28_*[A-Z].txt | awk '{print $7}' > tempt/28.txt
grep CAUT 29_*[A-Z].txt | awk '{print $7}' > tempt/29.txt
grep CAUT 30_*[A-Z].txt | awk '{print $7}' > tempt/30.txt
grep CAUT 31_*[A-Z].txt | awk '{print $7}' > tempt/31.txt
grep CAUT 32_*[A-Z].txt | awk '{print $7}' > tempt/32.txt
grep CAUT 33_*[A-Z].txt | awk '{print $7}' > tempt/33.txt
grep CAUT 34_*[A-Z].txt | awk '{print $7}' > tempt/34.txt
grep CAUT 35_*[A-Z].txt | awk '{print $7}' > tempt/35.txt



echo "%, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42" > res.csv
echo "%, leukocyte, heartwall, cfd, lud, hotspot, backpropagation, nw, kmeans, bfs, srad, streamcluster, particlefilter, pathfinder, gaussian, nn, lavamd, myocyte, b+tree, gpudwt, hybridsort, 2DCONV, 3DCONV, ATAX, COVAR, GEMM, GRAMSCHM, SYR2K, 2MM, 3MM, BICG, CORR, FDTD-2D, GESUMMV, MVT, SYRK, histo(s), lbm(s), mri-gridding(s), mri-q(s), sad(s), sgemm(s), spmv(s)" >> res.csv
paste -d ,  tempt/0.txt tempt/1.txt tempt/2.txt tempt/3.txt tempt/4.txt tempt/5.txt tempt/6.txt tempt/7.txt tempt/8.txt tempt/9.txt tempt/10.txt tempt/11.txt tempt/12.txt tempt/13.txt tempt/14.txt tempt/15.txt tempt/16.txt tempt/17.txt tempt/18.txt tempt/19.txt  tempt/20.txt  tempt/21.txt  tempt/22.txt  tempt/23.txt tempt/24.txt  tempt/25.txt  tempt/26.txt  tempt/27.txt  tempt/28.txt  tempt/29.txt  tempt/30.txt   tempt/31.txt  tempt/32.txt  tempt/33.txt  tempt/34.txt tempt/35.txt $result_parboil_dir/tempt/histo.txt $result_parboil_dir/tempt/lbm.txt  $result_parboil_dir/tempt/mri-gridding.txt $result_parboil_dir/tempt/mri-q.txt $result_parboil_dir/tempt/sad.txt $result_parboil_dir/tempt/sgemm.txt $result_parboil_dir/tempt/spmv.txt >> res.csv


           


echo
echo
echo "Finished! Please look at the res.csv using EXCEL."
echo "The first column is the partitioning ratio. The time in col:1-35 is in ms while 36-42 in second."
