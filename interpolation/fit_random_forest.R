library(randomForest)
source("helpers.R")
load("data/EEA_sub_train2_aux.RData")
load("data/EEA_sub_test2_aux.RData")
load("data/EEA_sub_val2_aux.RData")

aux_var_names <- c("co", "nh3", "no2", "no", "o3", "pm10", "pm25", "so2", "voc", "rh", "lai_hv", "lai_lv", "ssr", "t2m", "tp", "winddir", "windspeed")
na_inds <- which(!complete.cases(EEA_sub_train2_aux[, aux_var_names]) | is.na(EEA_sub_train2_aux$mean_O3))
EEA_sub_train2_aux <- EEA_sub_train2_aux[-na_inds, ]

t_fit_start <- proc.time()
mean_o3_fit <- randomForest(mean_O3 ~ co + nh3 + no2 + no + o3 + pm10 + pm25 + so2 + voc + rh + ssr + t2m + tp + winddir + windspeed, data = EEA_sub_train2_aux, ntree = 50, importance = TRUE, na.action = na.exclude)
t_fit_end <- proc.time()
print(paste0("Random Forest fitting time (seconds): ", round(t_fit_end["elapsed"] - t_fit_start["elapsed"], 2)))
EEA_sub_train2_aux$mean_O3_res <- (EEA_sub_train2_aux$mean_O3) - predict(mean_o3_fit, newdata = EEA_sub_train2_aux)

na_inds_test <- which(!complete.cases(EEA_sub_test2_aux[, aux_var_names]) | is.na(EEA_sub_test2_aux$mean_O3))
EEA_sub_test2_aux <- EEA_sub_test2_aux[-na_inds_test, ]

t_inference_start <- proc.time()
test_preds <- predict(mean_o3_fit, newdata = EEA_sub_test2_aux)
EEA_sub_test2_aux$mean_O3_res <- (EEA_sub_test2_aux$mean_O3) - test_preds
t_inference_end <- proc.time()
print(paste0("Random Forest inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 2)))

na_inds_test <- which(!complete.cases(EEA_sub_test2_aux[, aux_var_names]) | is.na(EEA_sub_test2_aux$mean_O3))
EEA_sub_test2_aux <- EEA_sub_test2_aux[-na_inds_test, ]
EEA_sub_test2_aux$mean_O3_res <- (EEA_sub_test2_aux$mean_O3) - predict(mean_o3_fit, newdata = EEA_sub_test2_aux)

mae_rf <- mean(abs(EEA_sub_test2_aux$mean_O3_res), na.rm = TRUE)
rmse_rf <- sqrt(mean((EEA_sub_test2_aux$mean_O3_res)^2, na.rm = TRUE))
print(paste0("Random Forest MAE: ", mae_rf))
print(paste0("Random Forest RMSE: ", rmse_rf))