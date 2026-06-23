library(gstat)
library(sp)
library(spacetime)
library(sf)
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

load("data/EEA_sub_train_aux.RData")
load("data/EEA_sub_train2_aux.RData")
load("data/EEA_sub_test2_aux.RData")

n_train <- nrow(EEA_sub_train_aux)
set.seed(123)
n_val <- round(0.05 * n_train)
n_test <- round(0.05 * n_train)
val_inds <- sample(1:n_train, n_val)
test_inds <- sample(setdiff(1:n_train, val_inds), n_test)
train_inds <- setdiff(1:n_train, c(val_inds, test_inds))
EEA_sub_train2 <- EEA_sub_train_aux
EEA_sub_train2[val_inds, "mean_O3"] <- NA
EEA_sub_train2[test_inds, "mean_O3"] <- NA

coords_train_utm <- to_utm(EEA_sub_train2$Longitude, EEA_sub_train2$Latitude)
EEA_sub_train2$X <- coords_train_utm[,1]
EEA_sub_train2$Y <- coords_train_utm[,2]

all(na.omit(EEA_sub_train2_aux$mean_O3) == na.omit(EEA_sub_train2$mean_O3))

t_fit_start <- proc.time()
mean_o3_fit <- lm(mean_O3 ~ co + nh3 + no2 + no + o3 + pm10 + pm25 + so2 + voc + rh + ssr + lai_lv + lai_hv + t2m + tp + winddir + windspeed, data = EEA_sub_train2)
EEA_sub_train2$mean_O3_res <- EEA_sub_train2$mean_O3 - predict(mean_o3_fit, newdata = EEA_sub_train2)
EEA_sub_test2$mean_O3_res <- EEA_sub_test2$mean_O3 - predict(mean_o3_fit, newdata = EEA_sub_test2)

time_window <- length(unique(EEA_sub_train2$time_numeric))

var_df <- EEA_sub_train2[, c("AirQualityStation", "X", "Y", "time_numeric", "time", "mean_O3_res")]
stfdf_data <- create_st_dataset(var_df, c("mean_O3_res"), length.out = time_window)
summary(stfdf_data)
vv_var <- variogramST(mean_O3_res ~ 1, stfdf_data, tlags = 0:20, cutoff = 100000)
vv_var[1, 2:3] <- 0
plot(vv_var, wireframe = TRUE, main = "mean_O3_res")
var_res <- var(stfdf_data[, , "mean_O3_res"]@data[[1]], na.rm = TRUE)
sepindex(vario_st = vv_var, nt = 21, ns = 8, globalSill = var_res)
vg_obj <- fit_product_sum(vv_var, var_res, verbose = TRUE, init_t_sill = var_res, init_s_sill = var_res)
t_fit_end <- proc.time()
print(paste0("Regression+Kriging fitting time (seconds): ", round(t_fit_end["elapsed"] - t_fit_start["elapsed"], 2)))

EEA_sub_test2 <- EEA_sub_test2[!is.na(EEA_sub_test2$mean_O3_res), ]

stfdf_validation <- create_sti_dataset(EEA_sub_test2, "mean_O3_res")
t_inference_start <- proc.time()
pred_var_time <- krigeST(mean_O3_res ~ 1,
    data = stfdf_data,
    newdata = stfdf_validation, vg_obj$vgm_st, nmax = 200,
    stAni = 10000 / 4,
)
predictions_trend <- predict(mean_o3_fit, newdata = EEA_sub_test2)
t_inference_end <- proc.time()
print(paste0("Regression+Kriging inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 2)))
pred_var_df_time <- as.data.frame(pred_var_time)
pred_var_df_time <- pred_var_df_time[order(pred_var_df_time[, "sp.ID"]), ]
combined_preds <- pred_var_df_time$var1.pred + predictions_trend
mae_kriging <- mean(abs(EEA_sub_test2$mean_O3 - combined_preds), na.rm = TRUE)
rmse_kriging <- sqrt(mean((EEA_sub_test2$mean_O3 - combined_preds)^2, na.rm = TRUE))
print(paste0("Kriging MAE: ", mae_kriging))
print(paste0("Kriging RMSE: ", rmse_kriging))
inds1 <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 1))
inds10 <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 10))
bootstrap_results_1 <- bootstrap_errors(combined_preds[inds1], EEA_sub_test2$mean_O3[inds1], n_bootstrap = 1000, seed = 29092025)
bootstrap_results_10 <- bootstrap_errors(combined_preds[inds10], EEA_sub_test2$mean_O3[inds10], n_bootstrap = 1000, seed = 29092025)
print("Bootstrap results (Regression+Kriging 1-step-ahead):")
print(bootstrap_results_1)
print("Bootstrap results (Regression+Kriging 10-steps-ahead):")
print(bootstrap_results_10)
