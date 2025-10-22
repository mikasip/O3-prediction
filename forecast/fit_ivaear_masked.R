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

load("O3_prediction/data/EEA_sub_train_aux.RData")
load("O3_prediction/data/EEA_sub_val_aux.RData")
load("O3_prediction/data/EEA_sub_test.RData")
val_names <- colnames(EEA_sub_val_aux)
EEA_sub_train_aux <- EEA_sub_train_aux[, val_names]
EEA_sub_train_aux <- rbind(EEA_sub_train_aux, EEA_sub_val_aux)
data_all_cc <- EEA_sub_train_aux[, c("mean_O3", "mean_NO2", "mean_PM10", "t2m", "rh", "voc")]
min_time <- min(as.Date(EEA_sub_train_aux$time)) #- 180 # Shift the time so that the last season in the training data and the test data align better.
EEA_sub_train_aux$time_numeric <- as.numeric(as.Date(EEA_sub_train_aux$time) - min_time)

coordinates_latlon <- SpatialPoints(cbind(EEA_sub_train_aux$Longitude, EEA_sub_train_aux$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))

# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_utm <- spTransform(coordinates_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))

# Add X and Y columns to your dataframe
EEA_sub_train_aux$X <- coordinates(coordinates_utm)[,1]
EEA_sub_train_aux$Y <- coordinates(coordinates_utm)[,2]

coords_time_cc <- as.matrix(EEA_sub_train_aux[, c("X", "Y", "time_numeric")])

n_s <- nrow(unique(coords_time_cc[, 1:2]))
seed <- 29092025
ivae_radial3 <- iVAEar_radial(
    as.matrix(data_all_cc), 
    as.matrix(coords_time_cc[, 1:2]), as.matrix(coords_time_cc[, 3]),
    latent_dim = 6,
    seasonal_period = 365.25,
    ar_order = 4,
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
EEA_sub_test$time_numeric <- as.numeric(as.Date(EEA_sub_test$time) - min_time)
EEA_sub_test2 <- EEA_sub_test

ar_order <- 4
max_time <- max(coords_time_cc[, 3])
coords_time_no_na <- coords_time_cc#[!(EEA_sub_train_aux$AirQualityStation %in% na_stations), ]
last_coords_time <- as.matrix(coords_time_no_na[which(coords_time_no_na[, 3] %in% ((max_time - (ar_order - 1)):max_time)), ])
st_ar1_test_time <- predict_coords_to_IC_ar(ivae_radial3, last_coords_time[, 1:2], last_coords_time[, 3], NULL, 
    as.matrix(EEA_sub_test2[, c("X", "Y")]), (EEA_sub_test2[, "time_numeric"]), NULL, get_trend = FALSE, get_ar_coefs = FALSE)
preds_ic_ivae <- st_ar1_test_time$preds

preds_ivae <- predict(ivae_radial3, preds_ic_ivae, IC_to_data = TRUE)
mean(abs(preds_ivae[, 1] - EEA_sub_test2$mean_O3), na.rm = TRUE)

min_time_test <- (min(EEA_sub_test2$time_numeric))
step1_ahead_inds <- which(EEA_sub_test2$time_numeric == min_time_test)
ivae_mae1 <- mean(abs((as.matrix(preds_ivae[step1_ahead_inds, 1]) - EEA_sub_test2[step1_ahead_inds, "mean_O3"])), na.rm = TRUE)
ivae_rmse1 <- sqrt(mean((as.matrix(preds_ivae[step1_ahead_inds, 1]) - EEA_sub_test2[step1_ahead_inds, "mean_O3"])^2, na.rm = TRUE))
ivae_mae1
ivae_rmse1
step10_ahead_inds <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 10))
ivae_mae10 <- mean(abs((as.matrix(preds_ivae[step10_ahead_inds, 1]) - EEA_sub_test2[step10_ahead_inds, "mean_O3"])), na.rm = TRUE)
ivae_rmse10 <- sqrt(mean((as.matrix(preds_ivae[step10_ahead_inds, 1]) - EEA_sub_test2[step10_ahead_inds, "mean_O3"])^2, na.rm = TRUE))
ivae_mae10
ivae_rmse10
