library(ECoST)
library(tensorflow)
library(NonlinearBSS)
library(gstat)
library(sp)
library(spacetime)
source("helpers.R")

var_names <- c("lai_hv", "lai_lv", "rh", "ssr", "t2m", "tp", "winddir", "windspeed")
var_names_pollution <- c("co", "nh3", "no2", "no", "o3", "pm10", "pm25", "so2", "voc")

load("data/EEA_sub_val2_aux.RData")
load("data/EEA_sub_test2_aux.RData")
load("data/EEA_sub_train2_aux.RData")
coords_train_utm <- to_utm(EEA_sub_train2_aux$Longitude, EEA_sub_train2_aux$Latitude)
EEA_sub_train2_aux$X <- coords_train_utm[,1]
EEA_sub_train2_aux$Y <- coords_train_utm[,2]

coords_val_utm <- to_utm(EEA_sub_val2_aux$Longitude, EEA_sub_val2_aux$Latitude)
EEA_sub_val2_aux$X <- coords_val_utm[,1]
EEA_sub_val2_aux$Y <- coords_val_utm[,2]

aux_var_names <- c(var_names, var_names_pollution)
na_inds <- which(!complete.cases(EEA_sub_train2_aux[, aux_var_names]) | is.na(EEA_sub_train2_aux$mean_O3))
na_inds_val <- which(!complete.cases(EEA_sub_val2_aux[, aux_var_names]) | is.na(EEA_sub_val2_aux$mean_O3))
data_all_cc_val <- EEA_sub_val2_aux[-na_inds_val, c("mean_O3")]
coords_time_cc_val <- as.matrix(EEA_sub_val2_aux[-na_inds_val, c("X", "Y", "time_numeric")])
aux_cc_val <- as.matrix(EEA_sub_val2_aux[-na_inds_val, aux_var_names])

data_all_cc <- EEA_sub_train2_aux[-na_inds, c("mean_O3", "mean_NO2", "mean_PM10")]
coords_time_cc <- as.matrix(EEA_sub_train2_aux[-na_inds, c("X", "Y", "time_numeric")])
aux_data_cc <- as.matrix(EEA_sub_train2_aux[-na_inds, aux_var_names])
min_time <- min(coords_time_cc[, 3])
coords_time_cc[, 3] <- coords_time_cc[, 3] - min_time
summary(coords_time_cc)
seed <- 26062025
t_fit_start <- proc.time()
fitted_dk_st <- deep_kriging_st(as.matrix(data_all_cc[, "mean_O3"]), aux_data_cc,
                                coords_time_cc[, 1:2], coords_time_cc[, 3], 
                                val_data = as.matrix(data_all_cc_val), val_aux_data = aux_cc_val,
                                val_spatial_locations = coords_time_cc_val[, 1:2], val_time_points = coords_time_cc_val[, 3],
                                epochs = 20,
                                spatial_basis = c(2, 9, 17, 37, 60), temporal_basis = c(9, 17, 37, 73, 211, 555),
                                spatial_kernel = "wendland",
                                activation = "relu", seed = seed,
                                batch_size = 64, lr_start = 0.001, lr_end = 0.0001, rbf_hidden_units = c(1024, 512, 256),
                                aux_hidden_units = c(256, 128),
                                validation_split = 0)
t_fit_end <- proc.time()
print(paste0("Deep Kriging fitting time (seconds): ", round(t_fit_end["elapsed"] - t_fit_start["elapsed"], 2)))

coords_test_utm <- to_utm(EEA_sub_test2_aux$Longitude, EEA_sub_test2_aux$Latitude)
EEA_sub_test2_aux$X <- coords_test_utm[,1]
EEA_sub_test2_aux$Y <- coords_test_utm[,2]

na_inds <- which(is.na(EEA_sub_test2_aux$mean_O3) | !complete.cases(EEA_sub_test2_aux[, aux_var_names]))
EEA_sub_test3 <- EEA_sub_test2_aux[-na_inds, ]
EEA_sub_test3[, "time_numeric"] <- EEA_sub_test3[, "time_numeric"] - min_time
coords_time_test2 <- as.matrix(EEA_sub_test3[, c("X", "Y", "time_numeric")])
aux_data_test <- as.matrix(EEA_sub_test3[, aux_var_names])
O3_data_test2 <- EEA_sub_test3[, "mean_O3"]

t_inference_start <- proc.time()
predictions <- predict(fitted_dk_st, aux_data_test, coords_time_test2[, 1:2], as.matrix(coords_time_test2[, 3]))
t_inference_end <- proc.time()
print(paste0("Deep Kriging inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 2)))
head(predictions)
mae_dk <- mean(abs(as.numeric(predictions) - O3_data_test2), na.rm = TRUE)
rmse_dk <- sqrt(mean((as.numeric(predictions) - O3_data_test2)^2, na.rm = TRUE))
print(paste0("Deep Kriging MAE: ", mae_dk))
print(paste0("Deep Kriging RMSE: ", rmse_dk))
inds1 <- which(coords_time_test2[, 3] < (min(coords_time_test2[, 3]) + 1))
inds10 <- which(coords_time_test2[, 3] < (min(coords_time_test2[, 3]) + 10))
bootstrap_results_1 <- bootstrap_errors(as.numeric(predictions)[inds1], O3_data_test2[inds1], n_bootstrap = 1000, seed = 29092025)
bootstrap_results_10 <- bootstrap_errors(as.numeric(predictions)[inds10], O3_data_test2[inds10], n_bootstrap = 1000, seed = 29092025)
print("Bootstrap results (Deep Kriging 1-step-ahead):")
print(bootstrap_results_1)
print("Bootstrap results (Deep Kriging 10-steps-ahead):")
print(bootstrap_results_10)
