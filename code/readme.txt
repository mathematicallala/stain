The SPCN method is based on MATLAB2020b.

Environment configuration for all code except SPCN method: Python3.9, PyCharm2021.2.3.

The main packages used in the code are: cv2, os, numpy, matplotlib and so on.

For the DCGMM method, the original code is based on TensorFlow 1.0, and we modify the
package to use TensorFlow 2.6.

For PF, SPCN and DCGMM, the code is publicly available, and the parameter determination 
is involved in the source code. Therefore, for these three methods, we carry out experiments 
according to the parameter setting of the source code.

Parameter Settings for CT, LP and RFCC methods are as follows:
1) There is no parameter setting in the original paper of CT method.
2) In the experimental process of LP method, the value of N of staining class is 2.
3) In the original paper of RFCC method, the weight coefficient ω, convergence condition ϵ 
and the difference between the highest and second highest membership degree δ are 
determined. In the original paper, the author sets ω=0.55, δ=0.1, and the convergence condition 
ϵ is not set. In our experiment, set ϵ=10^(-2). In addition, the setting of the initial value of this 
method is based on experience. Here, we take the initial value of the staining area as 4.6 and 5.0 
respectively.

The parameters of these methods are specified in the code.
