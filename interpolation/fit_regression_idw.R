source("interpolation/idw.R")

load("data/EEA_sub_train2_aux.RData")
load("data/EEA_sub_test2_aux.RData")

na_inds <- which(!complete.cases(EEA_sub_train2_aux[, aux_var_names]) | is.na(EEA_sub_train2_aux$mean_O3))
EEA_sub_train2_aux <- EEA_sub_train2_aux[-na_inds, ]

coordinates_latlon <- SpatialPoints(cbind(EEA_sub_train2_aux$Longitude, EEA_sub_train2_aux$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))
# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_utm <- spTransform(coordinates_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))
EEA_sub_train2_aux$X <- coordinates(coordinates_utm)[,1]
EEA_sub_train2_aux$Y <- coordinates(coordinates_utm)[,2]

coords_time <- EEA_sub_train2_aux[, c("X", "Y", "time_numeric")]
na_inds <- which(!complete.cases(EEA_sub_test2_aux[, aux_var_names]) | is.na(EEA_sub_test2_aux$mean_O3))
EEA_sub_test2_aux <- EEA_sub_test2_aux[-na_inds, ]

coordinates_test_latlon <- SpatialPoints(cbind(EEA_sub_test2_aux$Longitude, EEA_sub_test2_aux$Latitude),
                                   proj4string = CRS("+proj=longlat +datum=WGS84"))
# Transform to UTM Zone 32N (best for Northern Italy)
coordinates_test_utm <- spTransform(coordinates_test_latlon, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))
EEA_sub_test2_aux$X <- coordinates(coordinates_test_utm)[,1]
EEA_sub_test2_aux$Y <- coordinates(coordinates_test_utm)[,2]

coords_time_test2 <- EEA_sub_test2_aux[, c("X", "Y", "time_numeric")]

lm_fitted <- lm(mean_O3 ~ co + nh3 + no2 + no + o3 + pm10 + pm25 + so2 + voc + rh + lai_hv + lai_lv + ssr + t2m + tp + winddir + windspeed, data = EEA_sub_train2_aux, na.action = na.exclude)
pred <- predict(lm_fitted, newdata = EEA_sub_train2_aux)
data_res <- EEA_sub_train2_aux$mean_O3 - pred
data_seas <- pred
pred_test <- predict(lm_fitted, newdata = EEA_sub_test2_aux)
data_test_seas <- pred_test

idw_predictions_test <- spatiotemporal_idw(
    data_res,
    coords_time,
    coords_time_test2,
    spatial_power = 1,
    temporal_power = 30,
    max_dist = 0.5,
    max_time_diff = 10
)

mae_idw <- mean(abs(idw_predictions_test + data_test_seas - EEA_sub_test2_aux$mean_O3), na.rm = TRUE)
rmse_idw <- sqrt(mean((idw_predictions_test + data_test_seas - EEA_sub_test2_aux$mean_O3)^2, na.rm = TRUE))
print(paste0("IDW MAE: ", mae_idw))
print(paste0("IDW RMSE: ", rmse_idw))
