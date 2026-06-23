
bootstrap_errors <- function(predictions, truth, n_bootstrap = 1000, seed = 123) {
    set.seed(seed)
    n <- length(predictions)
    mae_boot  <- numeric(n_bootstrap)
    rmse_boot <- numeric(n_bootstrap)
    for (i in seq_len(n_bootstrap)) {
        idx          <- sample(n, replace = TRUE)
        errs         <- as.numeric(predictions)[idx] - truth[idx]
        mae_boot[i]  <- mean(abs(errs), na.rm = TRUE)
        rmse_boot[i] <- sqrt(mean(errs^2, na.rm = TRUE))
    }
    list(
        mae     = mean(abs(as.numeric(predictions) - truth), na.rm = TRUE),
        rmse    = sqrt(mean((as.numeric(predictions) - truth)^2, na.rm = TRUE)),
        mae_ci  = quantile(mae_boot,  c(0.025, 0.975)),
        rmse_ci = quantile(rmse_boot, c(0.025, 0.975))
    )
}

to_utm <- function(lon, lat) {
    pts_ll  <- SpatialPoints(cbind(lon, lat), proj4string = CRS("+proj=longlat +datum=WGS84"))
    pts_utm <- spTransform(pts_ll, CRS("+proj=utm +zone=32 +datum=WGS84 +units=m"))
    coordinates(pts_utm)
}

bootstrap_errors_spatial <- function(predictions, truth, stations,
                                     n_bootstrap = 1000, seed = 123) {
    set.seed(seed)
    unique_stations <- unique(stations)
    mae_boot  <- numeric(n_bootstrap)
    rmse_boot <- numeric(n_bootstrap)
    for (i in seq_len(n_bootstrap)) {
        sel <- sample(unique_stations, replace = TRUE)
        idx <- unlist(lapply(sel, function(s) which(stations == s)))
        errs         <- as.numeric(predictions)[idx] - truth[idx]
        mae_boot[i]  <- mean(abs(errs), na.rm = TRUE)
        rmse_boot[i] <- sqrt(mean(errs^2, na.rm = TRUE))
    }
    list(
        mae     = mean(abs(as.numeric(predictions) - truth), na.rm = TRUE),
        rmse    = sqrt(mean((as.numeric(predictions) - truth)^2, na.rm = TRUE)),
        mae_ci  = quantile(mae_boot,  c(0.025, 0.975)),
        rmse_ci = quantile(rmse_boot, c(0.025, 0.975))
    )
}