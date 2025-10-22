load("data/EEA_sub_train2_aux.RData")
load("data/EEA_sub_test2_aux.RData")

aux_var_names <- c("co", "nh3", "no2", "no", "o3", "pm10", "pm25", "so2", "voc", "rh", "lai_hv", "lai_lv", "ssr", "t2m", "tp", "winddir", "windspeed")
na_inds <- which(!complete.cases(EEA_sub_train2_aux[, aux_var_names]) | is.na(EEA_sub_train2_aux$mean_O3))
EEA_sub_train2_aux <- EEA_sub_train2_aux[-na_inds, ]

na_inds <- which(!complete.cases(EEA_sub_test2_aux[, aux_var_names]) | is.na(EEA_sub_test2_aux$mean_O3))
EEA_sub_test2_aux <- EEA_sub_test2_aux[-na_inds, ]

lm_fitted <- lm(mean_O3 ~ co + nh3 + no2 + no + o3 + pm10 + pm25 + so2 + voc + rh + lai_hv + lai_lv + ssr + t2m + tp + winddir + windspeed, data = EEA_sub_train2_aux, na.action = na.exclude)
pred_test <- predict(lm_fitted, newdata = EEA_sub_test2_aux)
mae_lm <- mean(abs(pred_test - EEA_sub_test2_aux$mean_O3), na.rm = TRUE)
rmse_lm <- sqrt(mean((pred_test - EEA_sub_test2_aux$mean_O3)^2, na.rm = TRUE))
mae_lm
rmse_lm
