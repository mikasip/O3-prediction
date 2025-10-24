library(ECoST)
library(tensorflow)
library(NonlinearBSS)
library(gstat)
library(sp)
library(spacetime)

var_names <- c("lai_hv", "lai_lv", "rh", "ssr", "t2m", "tp", "winddir", "windspeed")
var_names_pollution <- c("co", "nh3", "no2", "no", "o3", "pm10", "pm25", "so2", "voc")

load("data/EEA_sub_val2_aux.RData")
load("data/EEA_sub_test2_aux.RData")
load("data/EEA_sub_train2_aux.RData")
coordinates_train_latlon <- SpatialPoints(cbind(EEA_sub_train2_aux$Longitude, EEA_sub_train2_aux$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))

# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_train_utm <- spTransform(coordinates_train_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))

# Add X and Y columns to your dataframe
EEA_sub_train2_aux$X <- coordinates(coordinates_train_utm)[,1]
EEA_sub_train2_aux$Y <- coordinates(coordinates_train_utm)[,2]

aux_var_names <- c(var_names, var_names_pollution)
na_inds <- which(!complete.cases(EEA_sub_train2_aux[, aux_var_names]))

data_all_cc <- EEA_sub_train2_aux[-na_inds, c("mean_O3", "mean_NO2", "mean_PM10")]
coords_time_cc <- as.matrix(EEA_sub_train2_aux[-na_inds, c("X", "Y", "time_numeric")])
aux_data_cc <- as.matrix(EEA_sub_train2_aux[-na_inds, aux_var_names])
min_time <- min(coords_time_cc[, 3])
coords_time_cc[, 3] <- coords_time_cc[, 3] - min_time
summary(coords_time_cc)
seed <- 26062025
ivae_radial7 <- iVAE_radial_spatio_temporal(
    as.matrix(data_all_cc), 
    as.matrix(coords_time_cc[, 1:2]), as.matrix(coords_time_cc[, 3]),
    latent_dim = 3,
    aux_data = aux_data_cc,
    spatial_basis = c(2, 9, 17, 37, 60),
    temporal_basis = c(9, 17, 37, 73, 211, 777),
    aux_hidden_units = c(1024, 512, 256),
    activation = "leaky_relu",
    spatial_kernel = "wendland",
    epochs = 160,
    get_elbo = FALSE,
    batch_size = 64,
    lr_start = 0.0001,
    lr_end = 0.0001,
    seed = seed
)

coordinates_test_latlon <- SpatialPoints(cbind(EEA_sub_test2_aux$Longitude, EEA_sub_test2_aux$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))

# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_test_utm <- spTransform(coordinates_test_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))

# Add X and Y columns to your dataframe
EEA_sub_test2_aux$X <- coordinates(coordinates_test_utm)[,1]
EEA_sub_test2_aux$Y <- coordinates(coordinates_test_utm)[,2]

na_inds <- which(is.na(EEA_sub_test2_aux$mean_O3) | !complete.cases(EEA_sub_test2_aux[, aux_var_names]))
EEA_sub_test3 <- EEA_sub_test2_aux[-na_inds, ]
EEA_sub_test3[, "time_numeric"] <- EEA_sub_test3[, "time_numeric"] - min_time

preds2_ic <- predict_coords_to_IC(ivae_radial7, as.matrix(EEA_sub_test3[, c("X", "Y")]), EEA_sub_test3[, "time_numeric"], new_aux_data = as.matrix(EEA_sub_test3[, aux_var_names]))
preds_ivae <- predict(ivae_radial7, preds2_ic, IC_to_data = TRUE)

# MAE
ivae_mae <- mean(abs(preds_ivae[, 1] - EEA_sub_test3$mean_O3), na.rm = TRUE)

# RMSE
ivae_rmse <- sqrt(mean((preds_ivae[, 1] - EEA_sub_test3$mean_O3)^2, na.rm = TRUE))

print(paste0("iVAE MAE: ", ivae_mae))
print(paste0("iVAE RMSE: ", ivae_rmse))