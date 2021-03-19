# HandGP

Predicting synergism using Gaussian processes and Hand principle.

## Citation

This is supplementary code for the 2021 paper:

@article{shapovalova2021,

  title={Non-parametric synergy modeling with Gaussian processes},
  
  author={Shapovalova, Yuliya and Heskes, Tom and Dijkstra, Tjeerd},
  
  journal={},
  
  year={}
  
}

## Dependencies

Running the source code requires libaries numpy, pandas, matplotlib. 


For running experiments with MuSyC model we use [synergy](https://pypi.org/project/synergy/) library.

For running experiments with HandGP model we use [GPflow library](https://www.gpflow.org/), [TensorFlow](https://www.tensorflow.org/) and [TensorFlow probability](https://www.tensorflow.org/probability). We tested the code with the following versions of these libraries: GPflow v.2.0.0,  TensorFlow v2.1.0 and TensorFlow probability v0.9.0. Additional libraries for running HandGP source code: scipy, pymc3. 

## HandGP
Reproduces experiments with HandGP model. Figures from the paper are saved to HandGP/figures, estimates and confidence intervals of the hyperparameters in HandGP/results. 

## MuSyC
Reproduces experiments with MuSyC model. Code is based on the library synergy. Figures from the paper are saved to MuSyC/figures, estimates and confidence intervals of the hyperparameters in MuSyC/results. 

## MedianEffectModel
Reproduces results from Chou & Talalay "Quantitative analysis of dose-effect relationships:  the combined effects of multiple drugs or enzyme inhibitors.", Advances in enzyme regulation, 1984. Requires R libraries readr, tibble, ggplot2, dplyr.
