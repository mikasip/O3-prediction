library(ECoST)
library(tensorflow)
library(NonlinearBSS)
library(gstat)
library(sp)
library(spacetime)
source("helpers.R")

load("data/EEA_sub_val.RData")
load("data/EEA_sub_test.RData")
load("data/EEA_sub_train_val_aux_interpolated_ivae.RData")
EEA_sub_aux_interpolated <- EEA_sub_aux_train_interpolated_ivae

utm_test_coords <- to_utm(EEA_sub_test$Longitude, EEA_sub_test$Latitude)
EEA_sub_test$X <- utm_test_coords[,1]
EEA_sub_test$Y <- utm_test_coords[,2]

na_stations <- unique(EEA_sub_aux_interpolated$AirQualityStation[is.na(EEA_sub_aux_interpolated$mean_O3)])
na_station_inds <- which(EEA_sub_aux_interpolated$AirQualityStation %in% na_stations)
data_all_cc <- EEA_sub_aux_interpolated[-na_station_inds, c("mean_O3", "mean_NO2", "mean_PM10", "t2m", "rh", "voc")]
coords_time_cc <- as.matrix(EEA_sub_aux_interpolated[-na_station_inds, c("X", "Y", "time_numeric")])
min_time <- min(coords_time_cc[, 3])
coords_time_cc[, 3] <- coords_time_cc[, 3] - min_time
summary(coords_time_cc)
n_s <- nrow(unique(coords_time_cc[, 1:2]))
seed <- 29092025
t_fit_start <- proc.time()
ivae_radial <- iVAE_radial_spatio_temporal(
    as.matrix(data_all_cc), 
    as.matrix(coords_time_cc[, 1:2]), as.matrix(coords_time_cc[, 3]),
    latent_dim = 6,
    seasonal_period = 365.25,
    spatial_basis = c(2, 9, 17, 37),
    temporal_basis = c(9, 17, 37),
    aux_hidden_units = c(128),
    spatial_kernel = "wendland",
    #n_s = n_s,
    epochs = 60,
    get_elbo = FALSE,
    batch_size = 64,
    seed = seed
)
t_fit_end <- proc.time()
print(paste0("Fitting time (seconds): ", round(t_fit_end["elapsed"] - t_fit_start["elapsed"], 2)))

utm_test_coords <- to_utm(EEA_sub_test$Longitude, EEA_sub_test$Latitude)
EEA_sub_test$X <- utm_test_coords[,1]
EEA_sub_test$Y <- utm_test_coords[,2]

EEA_sub_test2 <- EEA_sub_test[!(EEA_sub_test$AirQualityStation %in% na_stations), ]
# Use only first 10 time steps of test data for inference
first10_cut <- min(EEA_sub_test$time_numeric) + 10
EEA_sub_test2 <- EEA_sub_test2[EEA_sub_test2$time_numeric < first10_cut, ]

t_inference_start <- proc.time()
preds_ic <-predict_coords_to_IC(ivae_radial, as.matrix(EEA_sub_test2[, c("X", "Y")]), EEA_sub_test2[, "time_numeric"] - min_time)
preds_ivae <- predict(ivae_radial, preds_ic, IC_to_data = TRUE)
t_inference_end <- proc.time()
print(paste0("Inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 3)))
mean(abs(preds_ivae[, 1] - EEA_sub_test2$mean_O3), na.rm = TRUE)
inds1 <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 1))
inds10 <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 10))
ivae_mae1 <- mean(abs((as.matrix(preds_ivae[inds1, 1]) - EEA_sub_test2[inds1, "mean_O3"])), na.rm = TRUE)
ivae_mae10 <- mean(abs((as.matrix(preds_ivae[inds10, 1]) - EEA_sub_test2[inds10, "mean_O3"])), na.rm = TRUE)
ivae_rmse1 <- sqrt(mean((as.matrix(preds_ivae[inds1, 1]) - EEA_sub_test2[inds1, "mean_O3"])^2, na.rm = TRUE))
ivae_rmse10 <- sqrt(mean((as.matrix(preds_ivae[inds10, 1]) - EEA_sub_test2[inds10, "mean_O3"])^2, na.rm = TRUE))
print(paste0("iVAE MAE (1-step-ahead): ", ivae_mae1))
print(paste0("iVAE MAE (10-steps-ahead): ", ivae_mae10))
print(paste0("iVAE RMSE (1-step-ahead): ", ivae_rmse1))
print(paste0("iVAE RMSE (10-steps-ahead): ", ivae_rmse10))
bootstrap_results_1 <- bootstrap_errors(preds_ivae[inds1, 1], EEA_sub_test2[inds1, "mean_O3"], n_bootstrap = 1000, seed = seed)
bootstrap_results_10 <- bootstrap_errors(preds_ivae[inds10, 1], EEA_sub_test2[inds10, "mean_O3"], n_bootstrap = 1000, seed = seed)
print("Bootstrap results (iVAE 1-step-ahead):")
print(bootstrap_results_1)
print("Bootstrap results (iVAE 10-steps-ahead):")
print(bootstrap_results_10)
