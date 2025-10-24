source("interpolation/idw.R")

load("data/EEA_sub_train2_aux.RData")
load("data/EEA_sub_val2_aux.RData")
load("data/EEA_sub_test2_aux.RData")

na_inds <- which(is.na(EEA_sub_train2_aux$mean_O3))

coordinates_latlon <- SpatialPoints(cbind(EEA_sub_train2_aux$Longitude, EEA_sub_train2_aux$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))
# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_utm <- spTransform(coordinates_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))
# Add X and Y columns to your dataframe
EEA_sub_train2_aux$X <- coordinates(coordinates_utm)[,1]
EEA_sub_train2_aux$Y <- coordinates(coordinates_utm)[,2]

coordinates_test_latlon <- SpatialPoints(cbind(EEA_sub_test2_aux$Longitude, EEA_sub_test2_aux$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))
# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_test_utm <- spTransform(coordinates_test_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))
# Add X and Y columns to your dataframe
EEA_sub_test2_aux$X <- coordinates(coordinates_test_utm)[,1]
EEA_sub_test2_aux$Y <- coordinates(coordinates_test_utm)[,2]

O3_data_train <- EEA_sub_train2_aux[-na_inds, "mean_O3"]
coords_time <- EEA_sub_train2_aux[-na_inds, c("X", "Y", "time_numeric")]
coords_time_test <- EEA_sub_test2_aux[, c("X", "Y", "time_numeric")]
data <- O3_data_train
xc <- cos(2 * pi * coords_time[, 3] / 365)
xs <- sin(2 * pi * coords_time[, 3] / 365)
data_df <- as.data.frame(cbind(data, xc, xs))
names(data_df) <- c("mean_O3", "xc", "xs")
fitlm <- lm(mean_O3 ~ xc + xs, data = data_df)
pred <- predict(fitlm, newdata = data_df)
data_res <- data_df$mean_O3 - pred
data_seas <- pred
xc_test <- cos(2 * pi * coords_time_test[, 3] / 365)
xs_test <- sin(2 * pi * coords_time_test[, 3] / 365)
data_test_seas <- predict(fitlm, newdata = data.frame(xc = xc_test, xs = xs_test))

idw_predictions_test <- spatiotemporal_idw(
    data_res,
    coords_time,
    coords_time_test,
    spatial_power = 1,
    temporal_power = 30,
    max_dist = 0.5,
    max_time_diff = 10
)

mae_idw <- mean(abs(idw_predictions_test + data_test_seas - EEA_sub_test2_aux$mean_O3), na.rm = TRUE)
rmse_idw <- sqrt(mean((idw_predictions_test + data_test_seas - EEA_sub_test2_aux$mean_O3)^2, na.rm = TRUE))
print(paste0("IDW MAE: ", mae_idw))
print(paste0("IDW RMSE: ", rmse_idw))
