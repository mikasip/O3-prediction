source("interpolation/idw.R")
source("helpers.R")

load("data/EEA_sub_train2_aux.RData")
load("data/EEA_sub_test2_aux.RData")

na_inds <- which(!complete.cases(EEA_sub_train2_aux[, aux_var_names]) | is.na(EEA_sub_train2_aux$mean_O3))
EEA_sub_train2_aux <- EEA_sub_train2_aux[-na_inds, ]

coords_train_utm <- to_utm(EEA_sub_train2_aux$Longitude, EEA_sub_train2_aux$Latitude)
EEA_sub_train2_aux$X <- coords_train_utm[,1]
EEA_sub_train2_aux$Y <- coords_train_utm[,2]

coords_time <- EEA_sub_train2_aux[, c("X", "Y", "time_numeric")]
na_inds <- which(!complete.cases(EEA_sub_test2_aux[, aux_var_names]) | is.na(EEA_sub_test2_aux$mean_O3))
EEA_sub_test2_aux <- EEA_sub_test2_aux[-na_inds, ]

coords_test_utm <- to_utm(EEA_sub_test2_aux$Longitude, EEA_sub_test2_aux$Latitude)
EEA_sub_test2_aux$X <- coords_test_utm[,1]
EEA_sub_test2_aux$Y <- coords_test_utm[,2]

coords_time_test2 <- EEA_sub_test2_aux[, c("X", "Y", "time_numeric")]

t_fit_start <- proc.time()
lm_fitted <- lm(mean_O3 ~ co + nh3 + no2 + no + o3 + pm10 + pm25 + so2 + voc + rh + lai_hv + lai_lv + ssr + t2m + tp + winddir + windspeed, data = EEA_sub_train2_aux, na.action = na.exclude)
pred <- predict(lm_fitted, newdata = EEA_sub_train2_aux)
data_res <- EEA_sub_train2_aux$mean_O3 - pred
data_seas <- pred
pred_test <- predict(lm_fitted, newdata = EEA_sub_test2_aux)
data_test_seas <- pred_test
t_fit_end <- proc.time()
print(paste0("Regression+IDW fitting time (seconds): ", round(t_fit_end["elapsed"] - t_fit_start["elapsed"], 2)))

t_inference_start <- proc.time()
idw_predictions_test <- spatiotemporal_idw(
    data_res,
    coords_time,
    coords_time_test2,
    spatial_power = 1,
    temporal_power = 30,
    max_dist = 0.5,
    max_time_diff = 10
)
t_inference_end <- proc.time()
print(paste0("Regression+IDW inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 2)))

predictions <- idw_predictions_test + data_test_seas
mae_idw <- mean(abs(predictions - EEA_sub_test2_aux$mean_O3), na.rm = TRUE)
rmse_idw <- sqrt(mean((predictions - EEA_sub_test2_aux$mean_O3)^2, na.rm = TRUE))
print(paste0("IDW MAE: ", mae_idw))
print(paste0("IDW RMSE: ", rmse_idw))
inds1 <- which(coords_time_test2[, 3] < (min(coords_time_test2[, 3]) + 1))
inds10 <- which(coords_time_test2[, 3] < (min(coords_time_test2[, 3]) + 10))
bootstrap_results_1 <- bootstrap_errors(predictions[inds1], EEA_sub_test2_aux$mean_O3[inds1], n_bootstrap = 1000, seed = 29092025)
bootstrap_results_10 <- bootstrap_errors(predictions[inds10], EEA_sub_test2_aux$mean_O3[inds10], n_bootstrap = 1000, seed = 29092025)
print("Bootstrap results (Regression+IDW 1-step-ahead):")
print(bootstrap_results_1)
print("Bootstrap results (Regression+IDW 10-steps-ahead):")
print(bootstrap_results_10)
