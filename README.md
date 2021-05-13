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
- Unsteady Burger's equation
  - `Burgers_Equation.py`: environment toggle name `Burger`. The formulation for this problem is shown below:
  ![Unsteady Burger's Equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26u_t&plus;uu_x%20%3D%20%5Cxi%20u_%7Bxx%7D%2C%20%5Cquad%20%5Ctext%7Bfor%20%24%28x%2Ct%29%5Cin%20%5B-1%2C1%5D%5Ctimes%5B0%2C1%5D%24%7D%5C%5C%20%26u%280%2Cx%29%20%3D%20-%5Csin%28%5Cpi%20x%29%2C%5C%5C%20%26u%28t%2C-1%29%20%3D%20u%28t%2C1%29%20%3D%200%20%5Cend%7Bsplit%7D)
- Nonlinear diffusion equation
- 1D convection-diffusion equation
- 2D convection-diffusion equation

