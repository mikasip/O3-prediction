source("interpolation/idw.R")
source("helpers.R")

load("data/EEA_sub_train2_aux.RData")
load("data/EEA_sub_val2_aux.RData")
load("data/EEA_sub_test2_aux.RData")

na_inds <- which(is.na(EEA_sub_train2_aux$mean_O3))

coords_train_utm <- to_utm(EEA_sub_train2_aux$Longitude, EEA_sub_train2_aux$Latitude)
EEA_sub_train2_aux$X <- coords_train_utm[,1]
EEA_sub_train2_aux$Y <- coords_train_utm[,2]

coords_test_utm <- to_utm(EEA_sub_test2_aux$Longitude, EEA_sub_test2_aux$Latitude)
EEA_sub_test2_aux$X <- coords_test_utm[,1]
EEA_sub_test2_aux$Y <- coords_test_utm[,2]

O3_data_train <- EEA_sub_train2_aux[-na_inds, "mean_O3"]
coords_time <- EEA_sub_train2_aux[-na_inds, c("X", "Y", "time_numeric")]
coords_time_test <- EEA_sub_test2_aux[, c("X", "Y", "time_numeric")]
data <- O3_data_train
xc <- cos(2 * pi * coords_time[, 3] / 365)
xs <- sin(2 * pi * coords_time[, 3] / 365)
data_df <- as.data.frame(cbind(data, xc, xs))
names(data_df) <- c("mean_O3", "xc", "xs")
fitlm <- lm(mean_O3 ~ xc + xs, data = data_df)
pred <- predict(fitlm, newdata = data_df)
data_res <- data_df$mean_O3 - pred
data_seas <- pred
xc_test <- cos(2 * pi * coords_time_test[, 3] / 365)
xs_test <- sin(2 * pi * coords_time_test[, 3] / 365)
data_test_seas <- predict(fitlm, newdata = data.frame(xc = xc_test, xs = xs_test))

t_inference_start <- proc.time()
idw_predictions_test <- spatiotemporal_idw(
    data_res,
    coords_time,
    coords_time_test,
    spatial_power = 1,
    temporal_power = 30,
    max_dist = 0.5,
    max_time_diff = 10
)
t_inference_end <- proc.time()
print(paste0("IDW inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 2)))

predictions <- idw_predictions_test + data_test_seas
mae_idw <- mean(abs(predictions - EEA_sub_test2_aux$mean_O3), na.rm = TRUE)
rmse_idw <- sqrt(mean((predictions - EEA_sub_test2_aux$mean_O3)^2, na.rm = TRUE))
print(paste0("IDW MAE: ", mae_idw))
print(paste0("IDW RMSE: ", rmse_idw))
inds1 <- which(coords_time_test[, 3] < (min(coords_time_test[, 3]) + 1))
inds10 <- which(coords_time_test[, 3] < (min(coords_time_test[, 3]) + 10))
bootstrap_results_1 <- bootstrap_errors(predictions[inds1], EEA_sub_test2_aux$mean_O3[inds1], n_bootstrap = 1000, seed = 29092025)
bootstrap_results_10 <- bootstrap_errors(predictions[inds10], EEA_sub_test2_aux$mean_O3[inds10], n_bootstrap = 1000, seed = 29092025)
print("Bootstrap results (IDW 1-step-ahead):")
print(bootstrap_results_1)
print("Bootstrap results (IDW 10-steps-ahead):")
print(bootstrap_results_10)
