Matlab functions for estimation (denoising/reconstruction) of approximately piecewise constant signals.
The functions are reference implementations of the method described in the paper

M. Storath, A. Weinmann, M. Unser. Exact algorithms for L^1-TV regularization of real-valued or circle-valued signals. 
SIAM Journal on Scientific Computing, 38(1), A614-A630, 2016

[Technical report here](https://arxiv.org/pdf/1504.00499.pdf)

[See also Pottslab for related functions](https://github.com/mstorath/Pottslab)

## Estimation of real-valued signals
Estimates a real-valued signal using the L1-TV model (exact non-iterative solver)

![L1 TV Denoising of real-valued signal](/Docs/L1TV_Real.png)

## Estimation of circle-valued signals
Estimates a circle-valued signal using the L1-TV model (exact non-iterative solver)

![L1 TV Denoising of circle-valued signal](/Docs/L1TV_Circ.png)

## Installation and usage
- Set the Matlab path by calling setPath.m
- Run the demos of the Demos folder
