> Description; 
> Creating a 2D gaussian Rasterizer which takes in a gaussian and optimizes them to get a 2D image.
> This is a side project which takes in account of doing diffusion on 2D splats and much more!

Tags: #yellow 

---


10-10-2024, Thursday

## Current Progress

- [x] Writing the code for rasterizing the Image from gaussians.
- [x] Writing a Training code for the optimization of the parameters for the gaussians.
- [x] Working on the `nan` values coming during the gradient updates
	- [x] Optimizing on the normalized images for better results. 

### Training code

 The `optimize` function takes care of the training of the gaussian parameters.
 The main  challenge in the code was figuring out how to prevent the `nan` values during the optimization, as well as prevent the floating point errors during the calculation of `covariance` matrix and `determinant`.
	 The `opacity` x `exp(-det)` was also causing the burst of `nan` values.
	 The main fix for that is to optimize for the <u>logarithmic</u> values of `l1, l2, l3` and hence exponentiate them during forward pass.

Things to do to never get the `nan` values and also initialize the initial parameter values in a proper sense!
- [ ] Initialize the values of the `cholesky_coeff` using `randn` instead of `rand` and now also trying `tanh` & `logit` activation for it.
- [ ] Checking if removing the `weight_decay` from` Adam` changes anything?
- [ ] Implementing the `SSIM` Loss function instead of `MSE` or a weight sum of both.
- [ ] When I increased the number of initial splats to 1000, I saw too many bright gaussians, hence too much overlapping, need to implement the removal to Gaussian to prevent this.
- [ ] Checking that `TODO`
- [ ] Implementing the splitting of Gaussian based up off the threshold of `norm` of gradient and gaussian etc, search for other methods as well.
- [ ] Read the codebase of Gaussian-Image in free time, try to see how is a typical paper type code written and try to learn from it.
- [ ] After all of this, if possible find major optimizations in the algorithm or techniques which might speed up the process.
- [ ] Then if possible Implement it in C++!!!!

## Current problems
- [ ] The `mean` is between `[-1,1 ]` which is maintained by the `tanh()` function, this is leading to no gaussian at the edge of the image.
- [ ] ![[Pasted image 20241217223611.png]]
- [ ] The fixed kernel size we are using is creating a problem, because parts where high detailing is required are unable to express themselves and hence require small kernels which in turn causes increase in number of splats to represent the whole scene.
- [ ] **We need to find some kind of method to represent gaussians in image coordinates without this kernel system.**

### Possible Speed-Ups
1. Mixed Precision Quantization: Particularly for the 3D gaussian Splats Models
2. Vector Quantization + Pruning: Mapping the common Gaussians  ![[Pasted image 20241216232957.png| 400]]
3. 


---
## References

- [Gaussian-Image | Github](https://github.com/Xinjie-Q/GaussianImage)
- [2D Gaussian Spalts | Github](https://github.com/OutofAi/2D-Gaussian-Splatting)
- [register_buffers and register_parameters](https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/10)
- [Efficient Gaussian Splats](https://www.youtube.com/watch?v=IUEzsWOOErE)
- [Paper Review | 3D Gaussian Splats](https://www.youtube.com/watch?v=2SQUMYw0h0A&t=14s)