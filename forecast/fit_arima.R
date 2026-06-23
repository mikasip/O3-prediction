# ARIMA (per-station, seasonal detrending) - Forecasting (O3)
# Uses train/val/test temporal splits (train up to 2022-05-31, test from 2023-05-31)
# Station-wise bootstrap: all time points preserved for sampled stations

library(forecast)

load("data/EEA_sub_train_val_aux_interpolated_ivae.RData")   # EEA_sub_aux_train_interpolated_ivae
load("data/EEA_sub_test.RData")
source("helpers.R")

# ---------- prepare data -----------------------------------------------------

train_df <- EEA_sub_aux_train_interpolated_ivae

# Drop stations that still have any NA in O3 after interpolation
na_stations <- unique(train_df$AirQualityStation[is.na(train_df$mean_O3)])
train_df    <- train_df[!train_df$AirQualityStation %in% na_stations, ]

test_df  <- EEA_sub_test[!EEA_sub_test$AirQualityStation %in% na_stations, ]

coords_time_train <- as.matrix(train_df[, c("Longitude", "Latitude", "time_numeric")])
coords_time_test  <- as.matrix(test_df[,  c("Longitude", "Latitude", "time_numeric")])

# ---------- fit & predict (ARIMA fitted once per station) --------------------

preds_all <- rep(NA_real_, nrow(test_df))

t_fit_start  <- proc.time()
t_pred_start_ref <- NULL   # set after last station fit
total_pred_time  <- 0

for (station in unique(train_df$AirQualityStation)) {

    train_idx <- which(train_df$AirQualityStation == station)
    test_idx  <- which(test_df$AirQualityStation  == station)
    if (length(test_idx) == 0) next

    # Seasonal detrending
    t_train <- coords_time_train[train_idx, 3]
    xc_tr   <- cos(2 * pi * t_train / 365.25)
    xs_tr   <- sin(2 * pi * t_train / 365.25)
    lm_seas <- lm(train_df$mean_O3[train_idx] ~ xc_tr + xs_tr)
    resids  <- residuals(lm_seas)

    t_test  <- coords_time_test[test_idx, 3]
    xc_te   <- cos(2 * pi * t_test / 365.25)
    xs_te   <- sin(2 * pi * t_test / 365.25)
    seas_te <- predict(lm_seas, newdata = data.frame(xc_tr = xc_te, xs_tr = xs_te))

    # ARIMA on residuals
    model <- auto.arima(resids)

    t0 <- proc.time()
    arima_preds <- predict(model, length(test_idx))$pred
    total_pred_time <- total_pred_time + (proc.time() - t0)["elapsed"]

    preds_all[test_idx] <- as.numeric(arima_preds) + seas_te
}
t_fit <- proc.time() - t_fit_start
# Subtract prediction time from fit time (they are interleaved per station)
t_fit_elapsed <- t_fit["elapsed"] - total_pred_time

cat("Fit time (s):", t_fit_elapsed, "\n")
cat("Inference time (s):", total_pred_time, "\n")

# ---------- evaluate ---------------------------------------------------------

truth   <- test_df$mean_O3
stations <- test_df$AirQualityStation

results_all <- bootstrap_errors_spatial(preds_all, truth, stations)

# 1-step and 10-step ahead subsets
min_t <- min(coords_time_test[, 3])
idx1  <- which(coords_time_test[, 3] < min_t + 1)
idx10 <- which(coords_time_test[, 3] < min_t + 10)

res1  <- bootstrap_errors_spatial(preds_all[idx1],  truth[idx1],  stations[idx1])
res10 <- bootstrap_errors_spatial(preds_all[idx10], truth[idx10], stations[idx10])

cat("\n=== ARIMA (O3 forecasting) ===\n")
cat(sprintf("All horizon  MAE = %.4f [%.4f, %.4f]  RMSE = %.4f [%.4f, %.4f]\n",
            results_all$mae, results_all$mae_ci[1], results_all$mae_ci[2],
            results_all$rmse, results_all$rmse_ci[1], results_all$rmse_ci[2]))
cat(sprintf("1-step ahead  MAE = %.4f [%.4f, %.4f]  RMSE = %.4f [%.4f, %.4f]\n",
            res1$mae, res1$mae_ci[1], res1$mae_ci[2],
            res1$rmse, res1$rmse_ci[1], res1$rmse_ci[2]))
cat(sprintf("10-step ahead MAE = %.4f [%.4f, %.4f]  RMSE = %.4f [%.4f, %.4f]\n",
            res10$mae, res10$mae_ci[1], res10$mae_ci[2],
            res10$rmse, res10$rmse_ci[1], res10$rmse_ci[2]))
cat(sprintf("Fit time (s): %.1f  |  Inference time (s): %.1f\n",
            t_fit_elapsed, total_pred_time))