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
ivae_radial3 <- NonlinearBSS:::iVAEar_radial(
    as.matrix(data_all_cc), 
    as.matrix(coords_time_cc[, 1:2]), as.matrix(coords_time_cc[, 3]),
    latent_dim = 6,
    seasonal_period = 365.25,
    ar_order = 3,
    spatial_basis = c(2, 9, 17, 37),
    temporal_basis = c(9, 17, 37),
    aux_hidden_units = c(64),
    spatial_kernel = "wendland",
    n_s = n_s,
    epochs = 70,
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
# Restrict inference to first 10 time steps of test data
first10_cut <- min(EEA_sub_test$time_numeric) + 10
EEA_sub_test2 <- EEA_sub_test2[EEA_sub_test2$time_numeric < first10_cut, ]

ar_order <- 3
max_time <- max(coords_time_cc[, 3])
last_coords_time <- as.matrix(coords_time_cc[which(coords_time_cc[, 3] %in% ((max_time - (ar_order - 1)):max_time)), ])
t_inference_start <- proc.time()
st_ar1_test_time <- predict_coords_to_IC_ar(ivae_radial3, last_coords_time[, 1:2], last_coords_time[, 3], NULL, 
    as.matrix(EEA_sub_test2[, c("X", "Y")]), (EEA_sub_test2[, "time_numeric"] - min_time), NULL, get_trend = TRUE, get_ar_coefs = TRUE)
preds_ic_ivae <- st_ar1_test_time$preds

preds_ivae <- predict(ivae_radial3, preds_ic_ivae, IC_to_data = TRUE)
t_inference_end <- proc.time()
print(paste0("Inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 2)))
mean(abs(preds_ivae[, 1] - EEA_sub_test2$mean_O3), na.rm = TRUE)

step1_ahead_inds <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 1))
ivae_mae1 <- mean(abs((as.matrix(preds_ivae[step1_ahead_inds, 1]) - EEA_sub_test2[step1_ahead_inds, "mean_O3"])), na.rm = TRUE)
ivae_rmse1 <- sqrt(mean((as.matrix(preds_ivae[step1_ahead_inds, 1]) - EEA_sub_test2[step1_ahead_inds, "mean_O3"])^2, na.rm = TRUE))

step10_ahead_inds <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 10))
ivae_mae10 <- mean(abs((as.matrix(preds_ivae[step10_ahead_inds, 1]) - EEA_sub_test2[step10_ahead_inds, "mean_O3"])), na.rm = TRUE)
ivae_rmse10 <- sqrt(mean((as.matrix(preds_ivae[step10_ahead_inds, 1]) - EEA_sub_test2[step10_ahead_inds, "mean_O3"])^2, na.rm = TRUE))
print(paste0("iVAE MAE (1-step-ahead): ", ivae_mae1))
print(paste0("iVAE RMSE (1-step-ahead): ", ivae_rmse1))
print(paste0("iVAE MAE (10-steps-ahead): ", ivae_mae10))
print(paste0("iVAE RMSE (10-steps-ahead): ", ivae_rmse10))
bootstrap_results_1 <- bootstrap_errors(preds_ivae[step1_ahead_inds, 1], EEA_sub_test2[step1_ahead_inds, "mean_O3"], n_bootstrap = 1000, seed = seed)
bootstrap_results_10 <- bootstrap_errors(preds_ivae[step10_ahead_inds, 1], EEA_sub_test2[step10_ahead_inds, "mean_O3"], n_bootstrap = 1000, seed = seed)
print("Bootstrap results (iVAE-AR 1-step-ahead):")
print(bootstrap_results_1)
print("Bootstrap results (iVAE-AR 10-steps-ahead):")
print(bootstrap_results_10)
