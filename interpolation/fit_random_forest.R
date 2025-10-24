library(randomForest)
load("data/EEA_sub_train2_aux.RData")
load("data/EEA_sub_test2_aux.RData")
load("data/EEA_sub_val2_aux.RData")

na_inds <- which(!complete.cases(EEA_sub_train2_aux[, aux_var_names]) | is.na(EEA_sub_train2_aux$mean_O3))
EEA_sub_train2_aux <- EEA_sub_train2_aux[-na_inds, ]

mean_o3_fit <- randomForest(mean_O3 ~ co + nh3 + no2 + no + o3 + pm10 + pm25 + so2 + voc + rh + ssr + t2m + tp + winddir + windspeed, data = EEA_sub_train2_aux, ntree = 50, importance = TRUE, na.action = na.exclude)
EEA_sub_train2_aux$mean_O3_res <- (EEA_sub_train2_aux$mean_O3) - predict(mean_o3_fit, newdata = EEA_sub_train2_aux)

na_inds_test <- which(!complete.cases(EEA_sub_test2_aux[, aux_var_names]) | is.na(EEA_sub_test2_aux$mean_O3))
EEA_sub_test2_aux <- EEA_sub_test2_aux[-na_inds_test, ]
EEA_sub_test2_aux$mean_O3_res <- (EEA_sub_test2_aux$mean_O3) - predict(mean_o3_fit, newdata = EEA_sub_test2_aux)

mae_rf <- mean(abs(EEA_sub_test2_aux$mean_O3_res), na.rm = TRUE)
rmse_rf <- sqrt(mean((EEA_sub_test2_aux$mean_O3_res)^2, na.rm = TRUE))
print(paste0("Random Forest MAE: ", mae_rf))
print(paste0("Random Forest RMSE: ", rmse_rf))