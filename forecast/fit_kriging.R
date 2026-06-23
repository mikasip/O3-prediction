library(gstat)
library(sp)
library(spacetime)
library(sf)
library(dplyr)
source("helpers.R")

fit_product_sum <- function(vv, res_var, verbose = FALSE, init_t_range = 10, init_s_range = 10000, init_t_sill = 1, init_s_sill = 1) {
    # Temporal variogram estimation
    vv_t_temp <- data.frame(vv)
    vv_t_temp <- vv_t_temp %>% filter(spacelag == 0)
    vv_t <- data.frame(cbind(vv_t_temp$np, as.numeric(vv_t_temp$timelag), vv_t_temp$gamma))
    vv_t <- vv_t[-1, ]
    names(vv_t) <- c("np", "dist", "gamma")
    class(vv_t) <- c("gstatVariogram", "data.frame")
    vg_t <- fit.variogram(vv_t, vgm(init_t_sill, c("Sph"), init_t_range, cutoff = 20))
    if (verbose) {
        print(vg_t)
        plot(x = vv_t_temp$timelag, y = vv_t_temp$gamma, main = "Temporal variogram", pch = 19)
    }
    vgm_t <- vgm(
        psill = vg_t$psill[1], model = vg_t$model[1],
        range = vg_t$range[1], nugget = 0,
        kappa = vg_t$kappa[1]
    )
    tm_mod <- variogramLine(
        vgm_t,
        dist_vector = seq(0, 20, by = 0.1)
    )
    if (verbose) lines(x = tm_mod[, 1], y = tm_mod[, 2], type = "l", col = "black")

    # Spatial variogram estimation
    vv_s_temp <- data.frame(vv)
    vv_s_temp <- vv_s_temp %>% filter(timelag == 0)
    vv_s <- data.frame(cbind(vv_s_temp$np, as.numeric(vv_s_temp$spacelag), vv_s_temp$gamma))
    vv_s <- vv_s[-1, ]
    names(vv_s) <- c("np", "dist", "gamma")
    class(vv_s) <- c("gstatVariogram", "data.frame")
    vg_s <- fit.variogram(vv_s, vgm(init_s_sill, c("Sph"), init_s_range))
    if (verbose) {
        print(vg_s)
        plot(x = vv_s_temp$spacelag, y = vv_s_temp$gamma, main = "Spatial variogram", pch = 19)
    }
    vgm_s <- vgm(
        psill = vg_s$psill[1], model = vg_s$model[1],
        range = vg_s$range[1], nugget = 0,
        kappa = vg_s$kappa[1]
    )
    s_mod <- variogramLine(
        vgm_s,
        dist_vector = seq(0, 3, by = 0.1)
    )
    if (verbose) lines(x = s_mod[, 1], y = s_mod[, 2], type = "l", col = "black")

    # compute the k parameter for the PRODUCT-SUM MODEL
    sillT <- res_var - vg_t$psill[1] # c00-sillsp
    #if (sillT < 0) sillT <- 0.01
    sillS <- res_var - vg_s$psill[1] # c00-sillt
    k <- (res_var - sillS - sillT) / (sillS * sillT)
    if (k < 0) k <- 0.001 # Fallback if k is negative
    print(sillT)
    print(sillS)
    print(k)

    # fit the PRODUCT-SUM MODEL
    vgm_prodSum_model <- vgmST("productSum", space = vgm_s, time = vgm_t, k = k)

    vgm_prodSum_fit <- fit.StVariogram(vv, vgm_prodSum_model, lower = c(0, 
                0.01, 0, 0, 0.01, 0, sqrt(.Machine$double.eps)))

    return(list(vgm_s = vgm_s, vgm_t = vgm_t, vgm_st = vgm_prodSum_fit))
}

create_st_dataset <- function(data, var_names, length.out) {
    sp <- cbind(data$X, data$Y)
    sp <- unique(sp)
    sp.names <- unique(data$AirQualityStation)
    colnames(sp) <- c("x", "y")
    startDate <- as.Date(min(data$time))
    sp2 <- sp::SpatialPoints(sp)
    row.names(sp2) <- sp.names
    time <- seq.Date(from = startDate, by = "day", length.out = length.out)
    ordered_df <- data[order(data[, "time"]), ]
    stfdf <- STFDF(sp = sp2, time = time, data = ordered_df[, var_names, drop = FALSE])
    return(stfdf)
}

create_sti_dataset <- function(data, var_names) {
    sp <- data[, c("X", "Y")]
    colnames(sp) <- c("x", "y")
    sp_points <- SpatialPoints(sp)
    row.names(sp_points) <- data$AirQualityStation
    time <- as.Date(data$time)
    
    stidf <- STIDF(sp = sp_points, time = time, data = data[, var_names, drop = FALSE])
    return(stidf)
}


load("data/EEA_sub_val.RData")
load("data/EEA_sub_test.RData")
load("data/EEA_sub_train_val_aux_interpolated_ivae.RData")

utm_test_coords <- to_utm(EEA_sub_test$Longitude, EEA_sub_test$Latitude)
EEA_sub_test$X <- utm_test_coords[,1]
EEA_sub_test$Y <- utm_test_coords[,2]

temp_df_train <- EEA_sub_aux_train_interpolated_ivae
na_stations <- unique(temp_df_train$AirQualityStation[is.na(temp_df_train$mean_O3)])
na_station_inds <- which(temp_df_train$AirQualityStation %in% na_stations)
temp_df_train <- temp_df_train[-na_station_inds, ]
n_s <- length(unique(temp_df_train$AirQualityStation))
n_t <- length(unique(temp_df_train$time_numeric))
n_s * n_t
temp_df_test <- EEA_sub_test
# Restrict kriging inference to first 10 time steps
first10_cut <- min(EEA_sub_test$time_numeric) + 10
temp_df_test <- temp_df_test[temp_df_test$time_numeric < first10_cut, ]
var <- "mean_O3"
xc <- cos(2 * pi * temp_df_train$time_numeric / 365)
xs <- sin(2 * pi * temp_df_train$time_numeric / 365)
xc_test <- cos(2 * pi * temp_df_test$time_numeric / 365)
xs_test <- sin(2 * pi * temp_df_test$time_numeric / 365)
temp_df_train <- cbind(temp_df_train, xc, xs)
temp_df_train$mean_O3_seas <- NA
temp_df_test$mean_O3_seas <- NA
test_xc_xs <- data.frame(xc = xc_test, xs = xs_test)
unique_stations_train <- unique(temp_df_train$AirQualityStation)
for (i in seq_along(unique_stations_train)) {
    station_inds <- which(temp_df_train$AirQualityStation == unique_stations_train[i])
    temp_1 <- temp_df_train[station_inds, ]
    n_t_station <- nrow(temp_1)
    fit.lm <- lm(mean_O3 ~ xc + xs, data = temp_1, na.action = na.exclude)
    fit <- fitted(fit.lm)
    temp_df_train[station_inds, ncol(temp_df_train)] <- fit
    station_inds_test <- which(temp_df_test$AirQualityStation == unique_stations_train[i])
    fit_test <- predict(fit.lm, newdata = test_xc_xs[station_inds_test, ])
    temp_df_test[station_inds_test, ncol(temp_df_test)] <- fit_test
}
temp_df_train$mean_O3_res <- temp_df_train$mean_O3 - temp_df_train$mean_O3_seas
temp_df_test$mean_O3_res <- temp_df_test$mean_O3 - temp_df_test$mean_O3_seas

max_time <- max(temp_df_train$time_numeric)
min_time <- min(temp_df_train$time_numeric)

var_df <- temp_df_train[, c("AirQualityStation", "X", "Y", "time_numeric", "time", "mean_O3_res")]
stfdf_data <- create_st_dataset(var_df, c("mean_O3_res"), length.out = n_t)
summary(stfdf_data)
t_fit_start <- proc.time()
vv_var <- variogramST(mean_O3_res ~ 1, stfdf_data, tlags = 0:20, cutoff = 100000)
vv_var[1, 2:3] <- 0
plot(vv_var, wireframe = TRUE, main = "mean_O3_res")
var_res <- var(stfdf_data[, , "mean_O3_res"]@data[[1]], na.rm = TRUE)
vg_obj <- fit_product_sum(vv_var, var_res, verbose = TRUE, init_t_sill = var_res, init_s_sill = var_res)
t_fit_end <- proc.time()
print(paste0("Kriging fitting time (seconds): ", round(t_fit_end["elapsed"] - t_fit_start["elapsed"], 2)))

first10_cut <- min(temp_df_test$time_numeric) + 10
temp_df_test <- temp_df_test[temp_df_test$time_numeric < first10_cut, ]
stfdf_pred <- create_sti_dataset(temp_df_test, "mean_O3_res")
t_inference_start <- proc.time()
pred_var_time <- krigeST(mean_O3_res ~ 1,
    data = stfdf_data,
    newdata = stfdf_pred, vg_obj$vgm_st, nmax = 200,
    stAni = 10000 / 8,
)
t_inference_end <- proc.time()
print(paste0("Kriging inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 2)))
pred_var_df_time <- as.data.frame(pred_var_time)
head(pred_var_df_time)
pred_var_df_time <- pred_var_df_time[order(pred_var_df_time[, "sp.ID"]), ]
mean(abs(pred_var_df_time$mean_O3_res), na.rm = TRUE)
mean(abs(pred_var_df_time$var1.pred - temp_df_test$mean_O3_res), na.rm = TRUE)

step1_ahead_inds <- which(temp_df_test$time_numeric < (min(temp_df_test$time_numeric) + 1))
kriging_mae1 <- mean(abs((as.matrix(temp_df_test[step1_ahead_inds, "mean_O3_res"]) - pred_var_df_time[step1_ahead_inds, "var1.pred"])), na.rm = TRUE)
step10_ahead_inds <- which(temp_df_test$time_numeric < (min(temp_df_test$time_numeric) + 10))
kriging_mae10 <- mean(abs((as.matrix(temp_df_test[step10_ahead_inds, "mean_O3_res"]) - pred_var_df_time[step10_ahead_inds, "var1.pred"])), na.rm = TRUE)
kriging_rmse1 <- sqrt(mean((as.matrix(temp_df_test[step1_ahead_inds, "mean_O3_res"]) - pred_var_df_time[step1_ahead_inds, "var1.pred"])^2, na.rm = TRUE))
kriging_rmse10 <- sqrt(mean((as.matrix(temp_df_test[step10_ahead_inds, "mean_O3_res"]) - pred_var_df_time[step10_ahead_inds, "var1.pred"])^2, na.rm = TRUE))
print(paste0("Kriging MAE (1-step-ahead): ", kriging_mae1))
print(paste0("Kriging RMSE (1-step-ahead): ", kriging_rmse1))
print(paste0("Kriging MAE (10-steps-ahead): ", kriging_mae10))
print(paste0("Kriging RMSE (10-steps-ahead): ", kriging_rmse10))
bootstrap_results_1 <- bootstrap_errors(pred_var_df_time[step1_ahead_inds, "var1.pred"], temp_df_test[step1_ahead_inds, "mean_O3_res"], n_bootstrap = 1000, seed = 29092025)
bootstrap_results_10 <- bootstrap_errors(pred_var_df_time[step10_ahead_inds, "var1.pred"], temp_df_test[step10_ahead_inds, "mean_O3_res"], n_bootstrap = 1000, seed = 29092025)
print("Bootstrap results (Kriging 1-step-ahead):")
print(bootstrap_results_1)
print("Bootstrap results (Kriging 10-steps-ahead):")
print(bootstrap_results_10)
