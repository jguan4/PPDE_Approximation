<!-- ---
permalink: /making-readmes-readable/
title: Making READMEs readable
--- -->
# PPDE Approximation: AMSC 663-664 Project
This project is designed to explore applications of machine learning methods in approximating parameter-dependent Partial Differential Equations (PDEs). 

This github repository holds the implementation needed to run POD-NN RB proposed in *non-intrusive reduced order modeling of nonlinear problems using neural networks* (2018) and PINN proposed in *physics-informed neural networks* (2019). 

The implementation is structured in a way such that each component is implemented relatively independently. Here is a general overview of what each component does:
- Driver: this component includes files `NN_Driver.py`, `DEMO.py` and `run_DEMO.py`. The file `NN_Driver.py` will call the environment script to create training and testing samples for different problems, call neural network scripts to initialize, train, reload or test weights and call optimizer scripts to optimize weights in the neural networks. `NN_Driver.py` will also distinguish the algorithm used (POD-NN RB or PINN) to run the appropriate algorithm. Files `DEMO.py` and `run_DEMO.py` call `NN_Driver.py` to input appropriate user-defined parameters.

- Environments: this component includes all files in `Environments` folder. Each script has functions that process training and testing samples, testing approximation produced by the network. The folder includes implementations for 
  - Viscous Burgers' Equation, 
  - Nonlinear Diffusion Equation, 
  - 1D and 2D Convection-Diffusion Equations. 

- Neural Networks: this component includes all files in the `NN` folder. Each script has functions that initialize and update weights, outputs gradient and Jacobian matrix for the weights and computes the loss. The folder includes implementations for 
  - Dense Net, 
  - ResNet and 
  - RNN. 

- Optimizers: this component includes all files in `Optimizers` folder. Each scripts has functions that compute updates to the weights and modify the weights of the network. The folder includes implementations for 
  - Levenberg-Marquardt 
  - Stochastic Gradient Descent
  - LBFGS


## Dependent Packages
The following packages are required to run the scripts:
```
- Tensorflow 2
- Numpy
- Scipy
- Matplotlib
- SMT
```

## Run Driver
In order to run `NN_Driver.py`, one needs to provide `DEMO.py` with appropriate inputs. `DEMO.py` requires 12 inputs and the last one is optional. Here are the inputs:
1. A string that represents the dimensions of the network. It is formatted so that it is a string of an array whose length is the number of hidden layers added by 1, and elements represent the width of the hidden layers. One example is '[32,32,32]', which represents a network with 2 hidden layers with 32 neurons per layer. 
2. A string that represents the sample sizes. It is formatted so that it is a string of an array with 4 entries. The first element represents the number of samples taken for the residual values; the second element represents the number of samples taken for the Dirchlet boundary conditions; the third element represents the number of samples taken for the Neumann boundary conditions; the fourth element represents the number of samples take for true solution or initial condition. One example for PINN is '[3000,600,300,0]', which represents a set of samples with 3000 samples for the residual value, 600 for the Dirchlet boundary conditions, 300 for the Neumann boundary condition and 0 for the initial condition. One example for POD-NN RB is '[0,0,0,500]', which represents a set of samples with 500 samples of parameters in the PDE problem. 
3. An integer that represents the number of testing solutions. 
4. An integer that represents the number of discretized intervals over the domain for generating testing solutions or snapshots in POD-NN RB.
5. A string formatted like an array containing a double that represents the regularization parameter in the loss function. This feature is not currently incorporated in the current code. One example is '[1e-7]'.  
6. An integer that serves as a toggle to choose the algorithm used. For 0, the driver will run POD-NN RB. For 1 or 2, the driver will run PINN. For 3, the driver will use RNN.
7. A string that selects the envrionment used. For "Poisson", the driver will test the nonlinear diffusion problem; for "CD_steady", the driver will test the 2D convection-diffusion problem; for "Burger", the driver will test the viscous Burgers' equation; for "CD_1D", the driver will test the 1D convection-diffusion problem.
8. A string that represents the weighting of each part in the loss function. This is more tailored for PINN. The string is formatted like an array with 4 entries, each one corresponding to the weighting of each type of the samples. One example is '[1,3,1,0]', which will weight the mean squared errors produced by Dirichlet boundary condition samples 3 times more heavier than the other types of samples. 
9. A string that serves as a toggle to select the type of networks used. For 'Dense', the driver will run a dense neural network; for 'Res', the driver will run a ResNet; for 'RNN', the driver will run a RNN. 
10. A string that serves as a toggle to select the optimizer used. For 'lm', the driver will use Leverberg-Marquardt; for 'sgd', the driver will use Stochastic Gradient Descent; for 'lbfgs', the driver will use L-BFGS. 
11. An integer that represents the dimension of the reduced basis L. If this entry is not provided, the driver will interpret L as 0. 

The easiest way to do so is to use `run_DEMO.py`. In `run_DEMO.py`, one could define the inputs as a list in the variable `combs`. One example is 
`[[32,32,32], [0,0,0,600], 50, 2048, [0], 0, "CD_1D", [0,0,0,1],"Res","lm",10]`
With this list, the driver will run POD-NN RB to solve the 1D convection-diffusion problem on a ResNet with 2 hidden layers with 32 neurons per layer. The number of samples is 600. The number of testing solutions is 50 and each solution is generated on a mesh of size 2048. The network will be optimized using Levernberg-Marquardt. The dimension of the reduced basis is 10.

### Train Weights
In `run_DEMO.py`, the variable `train_toggle` will determine if the driver will train or test the weights. To train a network and save the weights, set `train_toggle` to 1.

If there are existing weights and one wishes to restart training with the existing weights, set `train_toggle` to 2. 

The weights will be saved to the following file:
`./Log/Weights/{environment}/{algorithm}/Net{network structure}_Layers{network setup}_Ntr{sizes of samples}_h{mesh size}_Reg{regularization parameter}_Sample{input 6}_Weight{input 8}_Opt{optimizer toggle}_L{dimension of reduced basis}.npz`

### Load Trained Weights and Test
If one wishes to load trained weights to test and plot the approximations, set `train_toggle` to 3. 

### Environments
In the `Environments` folder, we provide many testing parameter-dependent PDE problems:
* Unsteady Burger's equation
  * `Burgers_Equation.py`: environment toggle name `Burger`. The formulation for this problem is shown below:

  ![Unsteady Burger's Equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26u_t&plus;uu_x%20%3D%20%5Cxi%20u_%7Bxx%7D%2C%20%5Cquad%20%5Ctext%7Bfor%20%24%28x%2Ct%29%5Cin%20%5B-1%2C1%5D%5Ctimes%5B0%2C1%5D%24%7D%5C%5C%20%26u%280%2Cx%29%20%3D%20-%5Csin%28%5Cpi%20x%29%2C%5C%5C%20%26u%28t%2C-1%29%20%3D%20u%28t%2C1%29%20%3D%200%20%5Cend%7Bsplit%7D)

* Nonlinear diffusion equation
  * `Poisson_1D.py`: environment toggle name `Poisson`. The formulation for this problem is shown below:

  ![Nonlinear Diffusion Equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26-%28exp%28u%28x%3B%5Cmathbf%7B%5Cxi%7D%29%29u%28x%3B%5Cmathbf%7B%5Cxi%7D%29%27%29%27%20%3D%20s%28x%3B%5Cmathbf%7B%5Cxi%7D%29%2C%20%5Cquad%20%5Ctext%7Bfor%20%24x%5Cin%20%28-%5Cfrac%7B%5Cpi%7D%7B2%7D%2C%5Cfrac%7B%5Cpi%7D%7B2%7D%29%24%7D%5C%5C%20%26u%28%5Cpm%20%5Cpi/2%3B%5Cmathbf%7B%5Cxi%7D%29%20%3D%20%5Cxi_2%5Csin%282%5Cpm%20%5Cfrac%7B%5Cxi_1%20%5Cpi%7D%7B2%7D%29exp%28%5Cpm%20%5Cfrac%7B%5Cxi_3%20%5Cpi%7D%7B2%7D%29%20%5Cend%7Bsplit%7D)

  where ![paremeter](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Cxi%7D%3D%28%5Cxi_1%2C%5Cxi_2%2C%5Cxi_3%29) are sampled on uniform distribution of [1,3] x [1,3] x [-0.5,0.5], and 

  ![sfunction](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20s%28x%3B%5Cmathbb%7B%5Cxi%7D%29%20%3D%26%20-%5Cxi_2exp%28%5Cxi_2%5Csin%282&plus;%5Cxi_1%20x%29exp%28%5Cxi_3%20x%29&plus;%5Cxi_3%20x%29%5C%5C%20%26*%5B2%5Cxi_1%5Cxi_3%5Ccos%282&plus;%5Cxi_1%20x%29&plus;%28%5Cxi_3%5E2-%5Cxi_1%5E2%29%5Csin%282&plus;%5Cxi_1%20x%29%5C%5C%20%26&plus;exp%28%5Cxi_3%20x%29%5B%5Cxi_1%5Ccos%282&plus;%5Cxi_1x%29&plus;%5Cxi_3%5Csin%282&plus;%5Cxi_1%20x%29%5D%5E2%5D%20%5Cend%7B%7D)

* 1D convection-diffusion equation
  We tested three different convection-diffusion equations and their transformed versions. 
  * `CD_1D_1.py`: environment toggle name `CD_1D_1`. The formulation for this problem is shown below: 

  ![1d convection-diffusion1](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26-%5Cepsilon%20u%27%27&plus;u%27%20%3D%201%2C%20%5Cquad%20x%5Cin%20%280%2C1%29%5C%5C%20%26u%280%29%3Du%281%29%3D0%20%5Cend%7B%7D)

  We tried many transformations on this problem. Let ![transform](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Cxi%3D%28a-x%29/%5Cepsilon%20%5Cend%7B%7D) and ![v](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20v%28%5Cxi%28x%29%29%3Du%28x%29%20%5Cend%7B%7D). 

  Then the transformed problem can be formulated as:

  ![transformed](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26-%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cfrac%7Bd%5E2v%7D%7Bd%5Cxi%5E2%7D-%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cfrac%7Bdv%7D%7Bd%5Cxi%7D%20%3D%201%2C%20%5Cquad%20%5Cxi%5Cin%20%28%5Cfrac%7Ba-1%7D%7B%5Cepsilon%7D%2C%5Cfrac%7Ba%7D%7B%5Cepsilon%7D%29%5C%5C%20%26v%28%5Cfrac%7Ba-1%7D%7B%5Cepsilon%7D%29%3Dv%28%5Cfrac%7Ba%7D%7B%5Cepsilon%7D%29%3D0%20%5Cend%7B%7D)

  Here are the transformed problem environments:
    * `CD_1D_2.py`: environment toggle name `CD_1D_2`. The transformation is for a=1.
    * `CD_1D_8.py`: environment toggle name `CD_1D_8`. The transformation is for a=0.5.
    * `CD_1D_9.py`: environment toggle name `CD_1D_9`. The transformation is for a=0.
    * `CD_1D_10.py`: environment toggle name `CD_1D_10`. The transformation is for a=0.25.
    * `CD_1D_11.py`: environment toggle name `CD_1D_11`. The transformation is for a=0.75.

  
  * Here is the second 1D convection-diffusion equation we tested. `CD_1D.py`: environment toggle name `CD_1D`. The formulation for this problem is shown below: 

  ![1d convection-diffusion2](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26-%5Cepsilon%20u%27%27&plus;u%27%20%3D%200%2C%20%5Cquad%20x%5Cin%20%280%2C1%29%5C%5C%20%26u%280%29%3Du%281%29%3D0%20%5Cend%7B%7D) 

    * `CD_1D_16.py`: environment toggle name `CD_1D_16`. We used the transformation of ![trans2](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Cxi%3D%281-x%29/%5Cepsilon%20%5Cend%7B%7D). The problem formulation becomes:

    ![trans2f](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20-%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cfrac%7Bd%5E2v%7D%7Bd%5Cxi%5E2%7D-%5Cfrac%7B1%7D%7B%5Cepsilon%7D%5Cfrac%7Bdv%7D%7Bd%5Cxi%7D%20%26%3D%200%20%5Cquad%20%5Ctext%7Bfor%20%7D%5Cxi%5Cin%20%280%2C%5Cfrac%7B1%7D%7B%5Cepsilon%7D%29%5C%5C%20v%280%29%20%26%3D%200%5C%5C%20v%281/%5Cepsilon%29%20%26%3D%201-e%5E%7B-1/%5Cepsilon%7D%5Cend%7B%7D)

  * `CD_1D_6.py`: environment toggle name `CD_1D_6`. This is the only 1D convection-diffusion problem with one Neumann boundary condition. The formulation for this problem is shown below: 
  ![1d convection-diffusion3](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26-%5Cepsilon%20u%27%27&plus;u%27%20%3D%201%2C%20%5Cquad%20x%5Cin%20%280%2C1%29%5C%5C%20%26u%280%29%3Du%27%281%29%3D0%20%5Cend%7B%7D)

    * `CD_1D_7.py`: environment toggle name `CD_1D_7`. We used the transformation of ![trans2](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Cxi%3D%281-x%29/%5Cepsilon%20%5Cend%7B%7D).

* 2D convection-diffusion equation
  We tested two different 2D convection-diffusion equations and transformed versions of them. 
  * `CD_2D_1.py`: environment toggle name `CD_2D_1`. The formulation for this problem is shown below: 

  ![2d convection-diffusion1](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26-%5Cepsilon%28u_%7Bxx%7D&plus;u_%7Byy%7D%29&plus;u_y%3D0%2C%20%5Cquad%20%28x%2Cy%29%5Cin%20%28-1%2C1%29%5Ctimes%20%28-1%2C1%29%5C%5C%20%26%20u%28-1%2Cy%29%20%5Capprox%20-1%20%2C%20%5Cquad%20u%281%2Cy%29%20%5Capprox%201%5C%5C%20%26%20u%28x%2C-1%29%20%3D%20x%2C%20%5Cquad%20u%28x%2C1%29%20%3D%200%20%5Cend%7B%7D)

    * `CD_2D_2.py`: environment toggle name `CD_2D_2`. We used the transformation ![2dtrans](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Cxi%20%3D%20x%2C%5Cquad%20%5Ceta%20%3D%20%281-y%29/%5Cepsilon%20%5Cend%7B%7D)

  * `CD_2D_5.py`: environment toggle name `CD_2D_5`. This problem has more than one boundary layer. The formulation for this problem is shown below: 

  ![2d convection-diffusion2](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26-%5Cepsilon%28u_%7Bxx%7D&plus;u_%7Byy%7D%29&plus;%281&plus;%28x&plus;1%29%5E2/4%29u_y%20%3D%200%2C%20%5Cquad%20%5Ctext%7Bfor%20%7D%28x%2Cy%29%5Cin%20%28-1%2C1%29%5Ctimes%20%28-1%2C1%29%5C%5C%20%26%20u%28-1%2Cy%29%20%3D%20%281-%281&plus;y%29/2%29%5E3%2C%20%5Cquad%20u%281%2Cy%29%20%3D%20%281-%281&plus;y%29/2%29%5E2%5C%5C%20%26%20u%28x%2C-1%29%20%3D%201%2C%20%5Cquad%20u_y%28x%2C1%29%20%3D%200%20%5Cend%7B%7D)
