# CoRunBench
Benchmark for Co-running Single Applications on Integrated Architectures

CoRunBench is based on Rodinia 3.0, Parboil and Polybench.
http://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Rodinia:Accelerating_Compute-Intensive_Applications_with_Accelerators
http://impact.crhc.illinois.edu/parboil/parboil.aspx
http://web.cse.ohio-state.edu/~pouchet/software/polybench/GPU/

1. You need to unzip the input data and make the directory "data" and the directory "CoRunBench" is the same directory.
You can use the following command:
tar xzvf data.tar.gz;
mv data ..;
2. Please set the OpenCL path in the file: common/make.config.
3. Run the script, run_all.sh, to get performance results.
Using this command:
bash run_all.sh;
4. Run the script, run_perfall.sh, to get performance results.
Using this command:
bash run_perfall.sh;
5. If you wants to try single application, please go into its directory and run related bash file.

Thanks for your interests in CoRunBench and hope you like it.

If you use this benchmark, please cite our paper:

Zhang F, Zhai J, Chen W, et al. To Co-Run, or Not To Co-Run: A Performance Study on Integrated Architectures[C]//Modeling, Analysis and Simulation of Computer and Telecommunication Systems (MASCOTS), 2015 IEEE 23rd International Symposium on. IEEE, 2015: 89-92.

Other programs will be released soon!

Feng
2016-Apr-24
