# README for manuscript code: Enhancing identifiable variational autoencoder in the presence of missing values and auxiliary covariates for spatio-temporal ozone modeling

This repository contains all code and data required to reproduce the results from the manuscript:
Sipilä, M., Cappello, C., De Iaco, S., Nordhausen, K., Palma, M., Taskinen, S., 2025. Enhancing identifiable variational autoencoder in the presence of missing values and auxiliary covariates for spatio-temporal ozone modeling. Manuscript

## Repository Structure
1. The subfolder `data` contains multivariate spatio-temporal datasets for training, validation, and testing the interpolation and forecasting tasks of ozone concentration fields in Northern Italy.

2. The subfolder `interpolation` includes scripts for training and evaluating different interpolation models, including
   - Liner regression
   - Regression kriging
   - Regression inverse distance weighting (IDW)
   - Random forest
   - Identifiable variational autoencoder (iVAE)
   - Deep kriging

3. The subfolder `forecast` includes scripts for training and evaluating different forecasting models, including
   - ARIMA
   - Kriging
   - iVAE
   - Autoregressive iVAE (iVAEar)
   - Masked iVAE
   - Masked iVAEar
   - Hierarchical spatio-temporal graph neural network (HiSTGNN)
  
## Reproduction Introductions

1. Clone the repository
```
git clone https://github.com/mikasip/O3-prediction.git
cd O3-prediction
```

2. Install required R packages
```
devtools::install_github("mikasip/ECoST")
devtools::install_github("mikasip/NonlinearBSS")
install.packages(c("forecast", "tensorflow", "keras3", "gstat", "sp", "spacetime", "sf", "randomForest"))
```

3. Run scripts (example)
```
source(interpolation/fit_linear_regression.R)
```
