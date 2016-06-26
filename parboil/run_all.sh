program_dir="/home/pacman/parboil/benchmarks"
result_dir="/home/pacman/parboil/result"

for i in histo  lbm  mri-gridding  mri-q  sad  sgemm  spmv  stencil  tpacf
do
  b=`echo "$i.txt"`
  echo $i $b
  cd $program_dir/$i
  bash run_zf.sh | tee $result_dir/$b
done


