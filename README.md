# CoRunBench
Benchmark for Co-running Single Applications on Integrated Architectures

CoRunBench is based on Rodinia 3.0, Parboil and Polybench.
http://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Rodinia:Accelerating_Compute-Intensive_Applications_with_Accelerators
http://impact.crhc.illinois.edu/parboil/parboil.aspx
http://web.cse.ohio-state.edu/~pouchet/software/polybench/GPU/

1. You need to unzip the input data "data.tar.gz" to get the directory "data", and move it with the directory "CoRunBench" in the same directory.
You can use the following command:
tar xzvf data.tar.gz;
mv data ..;
2. Please set the OpenCL path in the file: common/make.config.
3. Change the first and second lines of "run_all.sh".
program_dir is the root of CoRunBench and result_dir is the place you want to store the result.
Run the script, "run_all.sh", to get performance results.
Using this command:
bash run_all.sh;
4. Change the first and second lines of "run_perfall.sh".
program_dir is the root of CoRunBench and result_dir is the place you want to store the result. Run the script, "run_perfall.sh", to get performance results.
Using this command:
bash run_perfall.sh;
5. If you wants to try single application, please go into its directory and run related bash file.

If you use this benchmark, please cite our paper:

Zhang F, Zhai J, Chen W, et al. To Co-Run, or Not To Co-Run: A Performance Study on Integrated Architectures[C]//Modeling, Analysis and Simulation of Computer and Telecommunication Systems (MASCOTS), 2015 IEEE 23rd International Symposium on. IEEE, 2015: 89-92.

Thanks for your interests in CoRunBench and hope you like it.

Other programs will be released soon!

Feng
2016-Apr-24
