source("helpers.R")
load("data/EEA_sub_train2_aux.RData")
load("data/EEA_sub_test2_aux.RData")

aux_var_names <- c("co", "nh3", "no2", "no", "o3", "pm10", "pm25", "so2", "voc", "rh", "lai_hv", "lai_lv", "ssr", "t2m", "tp", "winddir", "windspeed")
na_inds <- which(!complete.cases(EEA_sub_train2_aux[, aux_var_names]) | is.na(EEA_sub_train2_aux$mean_O3))
EEA_sub_train2_aux <- EEA_sub_train2_aux[-na_inds, ]

na_inds <- which(!complete.cases(EEA_sub_test2_aux[, aux_var_names]) | is.na(EEA_sub_test2_aux$mean_O3))
EEA_sub_test2_aux <- EEA_sub_test2_aux[-na_inds, ]

t_fit_start <- proc.time()
lm_fitted <- lm(mean_O3 ~ co + nh3 + no2 + no + o3 + pm10 + pm25 + so2 + voc + rh + lai_hv + lai_lv + ssr + t2m + tp + winddir + windspeed, data = EEA_sub_train2_aux, na.action = na.exclude)
t_fit_end <- proc.time()
print(paste0("Linear Regression fitting time (seconds): ", round(t_fit_end["elapsed"] - t_fit_start["elapsed"], 2)))

t_inference_start <- proc.time()
pred_test <- predict(lm_fitted, newdata = EEA_sub_test2_aux)
t_inference_end <- proc.time()
print(paste0("Linear Regression inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 2)))

mae_lm <- mean(abs(pred_test - EEA_sub_test2_aux$mean_O3), na.rm = TRUE)
rmse_lm <- sqrt(mean((pred_test - EEA_sub_test2_aux$mean_O3)^2, na.rm = TRUE))
print(paste0("Linear Regression MAE: ", mae_lm))
print(paste0("Linear Regression RMSE: ", rmse_lm))
inds1 <- which(EEA_sub_test2_aux$time_numeric < (min(EEA_sub_test2_aux$time_numeric) + 1))
inds10 <- which(EEA_sub_test2_aux$time_numeric < (min(EEA_sub_test2_aux$time_numeric) + 10))
bootstrap_results_1 <- bootstrap_errors(pred_test[inds1], EEA_sub_test2_aux$mean_O3[inds1], n_bootstrap = 1000, seed = 29092025)
bootstrap_results_10 <- bootstrap_errors(pred_test[inds10], EEA_sub_test2_aux$mean_O3[inds10], n_bootstrap = 1000, seed = 29092025)
print("Bootstrap results (Linear Regression 1-step-ahead):")
print(bootstrap_results_1)
print("Bootstrap results (Linear Regression 10-steps-ahead):")
print(bootstrap_results_10)
