library(ECoST)
library(tensorflow)
library(NonlinearBSS)
library(gstat)
library(sp)
library(spacetime)

load("data/EEA_sub_train_aux.RData")
load("data/EEA_sub_val_aux.RData")
load("data/EEA_sub_test.RData")
val_names <- colnames(EEA_sub_val_aux)
EEA_sub_train_aux <- EEA_sub_train_aux[, val_names]
EEA_sub_train_aux <- rbind(EEA_sub_train_aux, EEA_sub_val_aux)
data_all_cc <- EEA_sub_train_aux[, c("mean_O3", "mean_NO2", "mean_PM10", "t2m", "rh", "voc")]
min_time <- min(as.Date(EEA_sub_train_aux$time)) 
EEA_sub_train_aux$time_numeric <- as.numeric(as.Date(EEA_sub_train_aux$time) - min_time)

coordinates_latlon <- SpatialPoints(cbind(EEA_sub_train_aux$Longitude, EEA_sub_train_aux$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))

# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_utm <- spTransform(coordinates_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))

# Add X and Y columns to your dataframe
EEA_sub_train_aux$X <- coordinates(coordinates_utm)[,1]
EEA_sub_train_aux$Y <- coordinates(coordinates_utm)[,2]

coords_time_cc <- as.matrix(EEA_sub_train_aux[, c("X", "Y", "time_numeric")])
seed <- 29092025
ivae_radial <- iVAE_radial_spatio_temporal(
    as.matrix(data_all_cc), 
    as.matrix(coords_time_cc[, 1:2]), as.matrix(coords_time_cc[, 3]),
    latent_dim = 6,
    seasonal_period = 365.25,
    spatial_basis = c(2, 9, 17, 37),
    temporal_basis = c(9, 17, 37),
    aux_hidden_units = c(128),
    spatial_kernel = "wendland",
    epochs = 60,
    get_elbo = FALSE,
    batch_size = 64,
    seed = seed
)

# Transform lat lon coords to X and Y:
coordinates_test_latlon <- SpatialPoints(cbind(EEA_sub_test$Longitude, EEA_sub_test$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))

# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_test_utm <- spTransform(coordinates_test_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))

# Add X and Y columns to your dataframe
EEA_sub_test$X <- coordinates(coordinates_test_utm)[,1]
EEA_sub_test$Y <- coordinates(coordinates_test_utm)[,2]
EEA_sub_test2 <- EEA_sub_test

preds_ic <- predict_coords_to_IC(ivae_radial, as.matrix(EEA_sub_test2[, c("X", "Y")]), EEA_sub_test2[, "time_numeric"] - as.numeric(min_time))
preds_ivae <- predict(ivae_radial, preds_ic, IC_to_data = TRUE)
mean(abs(preds_ivae[, 1] - EEA_sub_test2$mean_O3), na.rm = TRUE)
inds1 <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 1))
inds10 <- which(EEA_sub_test2$time_numeric < (min(EEA_sub_test2$time_numeric) + 10))
ivae_mae1 <- mean(abs((as.matrix(preds_ivae[inds1, 1]) - EEA_sub_test2[inds1, "mean_O3"])), na.rm = TRUE)
ivae_mae10 <- mean(abs((as.matrix(preds_ivae[inds10, 1]) - EEA_sub_test2[inds10, "mean_O3"])), na.rm = TRUE)
ivae_rmse1 <- sqrt(mean((as.matrix(preds_ivae[inds1, 1]) - EEA_sub_test2[inds1, "mean_O3"])^2, na.rm = TRUE))
ivae_rmse10 <- sqrt(mean((as.matrix(preds_ivae[inds10, 1]) - EEA_sub_test2[inds10, "mean_O3"])^2, na.rm = TRUE))
print(paste0("iVAE MAE (1-step-ahead): ", ivae_mae1))
print(paste0("iVAE MAE (10-steps-ahead): ", ivae_mae10))
print(paste0("iVAE RMSE (1-step-ahead): ", ivae_rmse1))
print(paste0("iVAE RMSE (10-steps-ahead): ", ivae_rmse10))

