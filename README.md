# CoRunBench
Benchmark for Co-running Single Applications on Integrated Architectures

##Introduction##
CoRunBench is a Benchmark for Co-running Single Applications on Integrated Architectures

CoRunBench is based on Rodinia 3.0, Parboil and Polybench.
*  [Rodinia](http://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Rodinia:Accelerating_Compute-Intensive_Applications_with_Accelerators)
*  [Parboil](http://impact.crhc.illinois.edu/parboil/parboil.aspx)
*  [Polybench](http://web.cse.ohio-state.edu/~pouchet/software/polybench/GPU/)

##Platform##
The current version of CoRunBench is implemented using the following platform.
* AMD A10-7850K
* Intel i7-4770R

##Guide##
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

##Publications##
If you use this benchmark, please cite our papers:

Feng Zhang, Jidong Zhai, Bingsheng He, Shuhao Zhang, Wenguang Chen. Understanding Co-running Behaviors on Integrated CPU/GPU Architectures. IEEE Trans. Parallel Distrib. Syst. (2016)
[bib](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7501903)

    @ARTICLE{7501903, 
     author={F. Zhang and J. Zhai and B. He and S. Zhang and W. Chen}, 
    journal={IEEE Transactions on Parallel and Distributed Systems}, 
    title={Understanding Co-running Behaviors on Integrated CPU/GPU Architectures}, 
year={2016}, 
volume={PP}, 
number={99}, 
pages={1-1}, 
keywords={Heterogeneous Computing;Integrated Architecture;Performance Prediction;Performance Tuning;Workload Characterization}, 
doi={10.1109/TPDS.2016.2586074}, 
ISSN={1045-9219}, 
month={},}

Feng Zhang, Jidong Zhai, Wenguang Chen, Bingsheng He, Shuhao Zhang. To Co-Run, or Not To Co-Run: A Performance Study on Integrated Architectures[C]//Modeling, Analysis and Simulation of Computer and Telecommunication Systems (MASCOTS), 2015 IEEE 23rd International Symposium on. IEEE, 2015: 89-92. [bib](https://scholar.google.com/scholar?hl=zh-CN&q=To+Co-Run%2C+or+Not+To+Co-Run%3A+A+Performance+Study+on+Integrated+Architectures&btnG=&lr=#)


##Acknowledgement##
*CoRunBench is developed by Tsinghua University, National University of Singapore.

Feng Zhang, Jidong Zhai and Wenguang Chen are with the Department of Computer Science and Technology, Tsinghua University, Beijing, 100084, China.

Bingsheng He and Shuhao Zhang are with the School of Computing, National University of Singapore, 119077, Singapore.


If you have problems, please contact:
* zhangfeng.thu.hpc@gmail.com
* tonyzhang19900609@gmail.com

Thanks for your interests in CoRunBench and hope you like it.
