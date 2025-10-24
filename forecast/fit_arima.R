library(forecast)

load("data/EEA_sub_val.RData")
load("data/EEA_sub_test.RData")
load("data/EEA_sub_train_val_aux_interpolated_ivae.RData")

preds_arima <- EEA_sub_test[, c("AirQualityStation", "time_numeric", "mean_O3")]
coords_time_test <- as.matrix(EEA_sub_test[, c("Longitude", "Latitude", "time_numeric")])
na_stations <- unique(EEA_sub_aux_train_interpolated_ivae$AirQualityStation[is.na(EEA_sub_aux_train_interpolated_ivae$mean_O3)])
na_station_inds <- which(EEA_sub_aux_train_interpolated_ivae$AirQualityStation %in% na_stations)
EEA_sub_aux_train_interpolated_ivae <- EEA_sub_aux_train_interpolated_ivae[-na_station_inds, ]
coords_time_train <- as.matrix(EEA_sub_aux_train_interpolated_ivae[, c("Longitude", "Latitude", "time_numeric")])

for (station in unique(EEA_sub_aux_train_interpolated_ivae$AirQualityStation)) {
    print(station)
    station_idxs <- which(EEA_sub_aux_train_interpolated_ivae$AirQualityStation == station)
    xc <- cos(2 * pi * coords_time_train[station_idxs, 3] / 365.25)
    xs <- sin(2 * pi * coords_time_train[station_idxs, 3] / 365.25)
    data_df <- as.data.frame(cbind(EEA_sub_aux_train_interpolated_ivae$mean_O3, xc, xs))
    names(data_df) <- c("mean_O3", "xc", "xs")
    fitlm <- lm(mean_O3 ~ xc + xs, data = data_df)
    pred <- predict(fitlm, newdata = data_df)
    data_res <- data_df$mean_O3 - pred
    data_seas <- pred
    test_station_idxs <- which(preds_arima$AirQualityStation == station)
    xc_test <- cos(2 * pi * coords_time_test[test_station_idxs, 3] / 365.25)
    xs_test <- sin(2 * pi * coords_time_test[test_station_idxs, 3] / 365.25)
    data_test_seas <- predict(fitlm, newdata = data.frame(xc = xc_test, xs = xs_test))
    model <- auto.arima(data_res)
    temp_preds <- predict(model, length(test_station_idxs))$pred
    temp_preds <- temp_preds + data_test_seas
    preds_arima[test_station_idxs, "mean_O3"] <- as.numeric(temp_preds)
}
arima_mae <- mean(abs((as.matrix(preds_arima[, "mean_O3"]) - EEA_sub_test[, "mean_O3"])), na.rm = TRUE)
inds1 <- which(coords_time_test[, 3] < min(coords_time_test[, 3]) + 1)
inds10 <- which(coords_time_test[, 3] < min(coords_time_test[, 3]) + 10)
# MAE
arima_mae1 <- mean(abs((as.matrix(preds_arima[inds10, "mean_O3"]) - EEA_sub_test[inds10, "mean_O3"])), na.rm = TRUE)
arima_mae10 <- mean(abs((as.matrix(preds_arima[inds1, "mean_O3"]) - EEA_sub_test[inds1, "mean_O3"])), na.rm = TRUE)
# RMSE
arima_rmse1 <- sqrt(mean((as.matrix(preds_arima[inds10, "mean_O3"]) - EEA_sub_test[inds10, "mean_O3"])^2, na.rm = TRUE))
arima_rmse10 <- sqrt(mean((as.matrix(preds_arima[inds1, "mean_O3"]) - EEA_sub_test[inds1, "mean_O3"])^2, na.rm = TRUE))
print(paste0("ARIMA MAE (1-step-ahead): ", arima_mae1))
print(paste0("ARIMA MAE (10-steps-ahead): ", arima_mae10))
print(paste0("ARIMA RMSE (1-step-ahead): ", arima_rmse1))
print(paste0("ARIMA RMSE (10-steps-ahead): ", arima_rmse10))