library(ECoST)
library(tensorflow)
library(NonlinearBSS)
library(gstat)
library(sp)
library(spacetime)
library(sf)
library(covatest)
library(dplyr)
library(rdist)

load("data/EEA_sub_val.RData")
load("data/EEA_sub_test.RData")
load("data/EEA_sub_train_val_aux_interpolated_ivae.RData")
EEA_sub_aux_interpolated <- EEA_sub_aux_train_interpolated_ivae

na_stations <- unique(EEA_sub_aux_interpolated$AirQualityStation[is.na(EEA_sub_aux_interpolated$mean_O3)])
na_station_inds <- which(EEA_sub_aux_interpolated$AirQualityStation %in% na_stations)
data_all_cc <- EEA_sub_aux_interpolated[-na_station_inds, c("mean_O3", "mean_NO2", "mean_PM10", "t2m", "rh", "voc")]
coords_time_cc <- as.matrix(EEA_sub_aux_interpolated[-na_station_inds, c("X", "Y", "time_numeric")])
min_time <- min(coords_time_cc[, 3]) - 180 # Shift the time so that the last season in the training data and the test data align better.
coords_time_cc[, 3] <- coords_time_cc[, 3] - min_time
summary(coords_time_cc)
n_s <- nrow(unique(coords_time_cc[, 1:2]))
seed <- 29092025
ivae_radial3 <- iVAEar_radial(
    as.matrix(data_all_cc), 
    as.matrix(coords_time_cc[, 1:2]), as.matrix(coords_time_cc[, 3]),
    latent_dim = 6,
    seasonal_period = 365.25,
    ar_order = 3,
    spatial_basis = c(2, 9, 17, 37),
    temporal_basis = c(9, 17, 37),
    aux_hidden_units = c(64),
    spatial_kernel = "wendland",
    n_s = n_s,
    epochs = 70,
    get_elbo = FALSE,
    batch_size = 64,
    seed = seed
)

# Transform lat lon coords to X and Y:
coordinates_val_latlon <- SpatialPoints(cbind(EEA_sub_test$Longitude, EEA_sub_test$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))

# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_val_utm <- spTransform(coordinates_val_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))

# Add X and Y columns to your dataframe
EEA_sub_test$X <- coordinates(coordinates_val_utm)[,1]
EEA_sub_test$Y <- coordinates(coordinates_val_utm)[,2]

EEA_sub_test2 <- EEA_sub_test[!(EEA_sub_test$AirQualityStation %in% na_stations), ]

ar_order <- 3
max_time <- max(coords_time_cc[, 3])
last_coords_time <- as.matrix(coords_time_cc[which(coords_time_cc[, 3] %in% ((max_time - (ar_order - 1)):max_time)), ])
st_ar1_test_time <- predict_coords_to_IC_ar(ivae_radial3, last_coords_time[, 1:2], last_coords_time[, 3], NULL, 
    as.matrix(EEA_sub_test2[, c("X", "Y")]), (EEA_sub_test2[, "time_numeric"] - min_time), NULL, get_trend = TRUE, get_ar_coefs = TRUE)
preds_ic_ivae <- st_ar1_test_time$preds

preds_ivae <- predict(ivae_radial3, preds_ic_ivae, IC_to_data = TRUE)
mean(abs(preds_ivae[, 1] - EEA_sub_test2$mean_O3), na.rm = TRUE)

step1_ahead_inds <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 1))
ivae_mae1 <- mean(abs((as.matrix(preds_ivae[step1_ahead_inds, 1]) - EEA_sub_test2[step1_ahead_inds, "mean_O3"])), na.rm = TRUE)
ivae_rmse1 <- sqrt(mean((as.matrix(preds_ivae[step1_ahead_inds, 1]) - EEA_sub_test2[step1_ahead_inds, "mean_O3"])^2, na.rm = TRUE))
ivae_mae1
ivae_rmse1
step10_ahead_inds <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 10))
ivae_mae10 <- mean(abs((as.matrix(preds_ivae[step10_ahead_inds, 1]) - EEA_sub_test2[step10_ahead_inds, "mean_O3"])), na.rm = TRUE)
ivae_rmse10 <- sqrt(mean((as.matrix(preds_ivae[step10_ahead_inds, 1]) - EEA_sub_test2[step10_ahead_inds, "mean_O3"])^2, na.rm = TRUE))
ivae_mae10
ivae_rmse10