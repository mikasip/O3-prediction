library(ECoST)
library(tensorflow)
library(NonlinearBSS)
library(gstat)
library(sp)
library(spacetime)

load("data/EEA_sub_val.RData")
load("data/EEA_sub_test.RData")
load("data/EEA_sub_train_val_aux_interpolated_ivae.RData")
EEA_sub_aux_interpolated <- EEA_sub_aux_train_interpolated_ivae

# Transform lat lon coords to X and Y:
coordinates_val_latlon <- SpatialPoints(cbind(EEA_sub_test$Longitude, EEA_sub_test$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))

# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_val_utm <- spTransform(coordinates_val_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))

# Add X and Y columns to your dataframe
EEA_sub_test$X <- coordinates(coordinates_val_utm)[,1]
EEA_sub_test$Y <- coordinates(coordinates_val_utm)[,2]

na_stations <- unique(EEA_sub_aux_interpolated$AirQualityStation[is.na(EEA_sub_aux_interpolated$mean_O3)])

na_station_inds <- which(EEA_sub_aux_interpolated$AirQualityStation %in% na_stations)
data_all_cc <- EEA_sub_aux_interpolated[-na_station_inds, c("mean_O3", "mean_NO2", "mean_PM10", "t2m", "rh", "voc")]
coords_time_cc <- as.matrix(EEA_sub_aux_interpolated[-na_station_inds, c("X", "Y", "time_numeric")])
cc_inds <- which(complete.cases(data_all_cc) == TRUE)
data_all_cc <- data_all_cc[cc_inds, ]
coords_time_cc <- coords_time_cc[cc_inds, ]

seq_in_len <- 20
seq_out_len <- 10
n_s <- length(unique(EEA_sub_aux_interpolated$AirQualityStation)) - length(na_stations)
model <- histgnn(
  data = as.matrix(data_all_cc),
  coords_time = coords_time_cc,
  seasonal_period = 365.25,
  neighborhood_radius = 20000,
  dropout = 0,
  seq_in_len = seq_in_len,
  seq_out_len = seq_out_len,
  device = 'cpu',
  gcn_depth = 2,
  stat_nodes = n_s,
  num_heads = 1,
  layers = 2,
  epochs = 40
)

na_station_inds_test <- which(EEA_sub_test$AirQualityStation %in% na_stations)
test_data10 <- EEA_sub_test[-na_station_inds_test, ]
test_data10 <- test_data10[test_data10$time_numeric < (min(test_data10$time_numeric) + 10), ]
coords_time_test <- as.matrix(test_data10[, c("X", "Y", "time_numeric")])
new_order <- order(coords_time_test[, 1], coords_time_test[, 2], coords_time_test[, 3])
rev_new_order <- order(new_order)

preds <- predict(model)
# save(preds, file = "data/histgnn_preds_test.RData")
# load("data/histgnn_preds_test.RData")
preds_ordered <- preds$predictions[rev_new_order, ]
colMeans(abs(preds_ordered[, 1:3] - test_data10[, c("mean_O3", "mean_NO2", "mean_PM10")]), na.rm = TRUE)
inds2 <- which(coords_time_test[,3] < (min(coords_time_test[,3]) + 1))
histgnn_mae1 <- mean(abs(preds_ordered[seq(1, nrow(preds_ordered), by = 10), 1] - test_data10[inds2, "mean_O3"]), na.rm = TRUE)
histgnn_mae10 <- mean(abs(preds_ordered[, 1] - test_data10[, "mean_O3"]), na.rm = TRUE)
histgnn_rmse10 <- sqrt(mean((preds_ordered[, 1] - test_data10[, "mean_O3"])^2, na.rm = TRUE))
histgnn_rmse1 <- sqrt(mean((preds_ordered[seq(1, nrow(preds_ordered), by = 10), 1] - test_data10[inds2, "mean_O3"])^2, na.rm = TRUE))
print(paste0("HISTGNN MAE (1-step-ahead): ", histgnn_mae1))
print(paste0("HISTGNN MAE (10-steps-ahead): ", histgnn_mae10))
print(paste0("HISTGNN RMSE (1-step-ahead): ", histgnn_rmse1))
print(paste0("HISTGNN RMSE (10-steps-ahead): ", histgnn_rmse10))
