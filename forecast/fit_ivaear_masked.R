library(ECoST)
library(tensorflow)
library(NonlinearBSS)
library(gstat)
library(sp)
library(spacetime)
source("helpers.R")

load("data/EEA_sub_train_aux.RData")
load("data/EEA_sub_val_aux.RData")
load("data/EEA_sub_test.RData")
load("data/EEA_sub_train_val_aux_interpolated_ivae.RData") # To get the NA stations to make fair comparisons with the other models (which are trained on the interpolated data).
val_names <- colnames(EEA_sub_val_aux)
EEA_sub_train_aux <- EEA_sub_train_aux[, val_names]
EEA_sub_train_aux <- rbind(EEA_sub_train_aux, EEA_sub_val_aux)
data_all_cc <- EEA_sub_train_aux[, c("mean_O3", "mean_NO2", "mean_PM10", "t2m", "rh", "voc")]
min_time <- min(as.Date(EEA_sub_train_aux$time))
EEA_sub_train_aux$time_numeric <- as.numeric(as.Date(EEA_sub_train_aux$time) - min_time)

utm_coords <- to_utm(EEA_sub_train_aux$Longitude, EEA_sub_train_aux$Latitude)
EEA_sub_train_aux$X <- utm_coords[,1]
EEA_sub_train_aux$Y <- utm_coords[,2]

coords_time_cc <- as.matrix(EEA_sub_train_aux[, c("X", "Y", "time_numeric")])

n_s <- nrow(unique(coords_time_cc[, 1:2]))
seed <- 03062026
t_fit_start <- proc.time()
ivae_radial3 <- iVAEar_radial(
    as.matrix(data_all_cc), 
    as.matrix(coords_time_cc[, 1:2]), as.matrix(coords_time_cc[, 3]),
    latent_dim = 5,
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
t_fit_end <- proc.time()
print(paste0("Fitting time (seconds): ", round(t_fit_end["elapsed"] - t_fit_start["elapsed"], 2)))

utm_test_coords <- to_utm(EEA_sub_test$Longitude, EEA_sub_test$Latitude)
EEA_sub_test$X <- utm_test_coords[,1]
EEA_sub_test$Y <- utm_test_coords[,2]
EEA_sub_test$time_numeric <- as.numeric(as.Date(EEA_sub_test$time) - min_time)
EEA_sub_test2 <- EEA_sub_test
# Restrict inference to first 10 time steps of test data
first10_cut <- min(EEA_sub_test2$time_numeric) + 10
EEA_sub_test2 <- EEA_sub_test2[EEA_sub_test2$time_numeric < first10_cut, ]

ar_order <- 4
max_time <- max(coords_time_cc[, 3])
coords_time_no_na <- coords_time_cc#[!(EEA_sub_train_aux$AirQualityStation %in% na_stations), ]
last_coords_time <- as.matrix(coords_time_no_na[which(coords_time_no_na[, 3] %in% ((max_time - (ar_order - 1)):max_time)), ])

t_inference_start <- proc.time()
st_ar1_test_time <- predict_coords_to_IC_ar(ivae_radial3, last_coords_time[, 1:2], last_coords_time[, 3], NULL, 
    as.matrix(EEA_sub_test2[, c("X", "Y")]), (EEA_sub_test2[, "time_numeric"]), NULL, get_trend = FALSE, get_ar_coefs = FALSE)
preds_ic_ivae <- st_ar1_test_time$preds
preds_ivae <- predict(ivae_radial3, preds_ic_ivae, IC_to_data = TRUE)
t_inference_end <- proc.time()
print(paste0("Inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 3)))
mean(abs(preds_ivae[, 1] - EEA_sub_test2$mean_O3), na.rm = TRUE)

na_stations <- unique(EEA_sub_aux_train_interpolated_ivae$AirQualityStation[is.na(EEA_sub_aux_train_interpolated_ivae$mean_O3)])
EEA_sub_test_filtered <- EEA_sub_test2[!(EEA_sub_test2$AirQualityStation %in% na_stations), ]
preds_ivae_filtered <- preds_ivae[!(EEA_sub_test2$AirQualityStation %in% na_stations), ]

min_time_test <- (min(EEA_sub_test_filtered$time_numeric))
step1_ahead_inds <- which(EEA_sub_test_filtered$time_numeric == min_time_test)
ivae_mae1 <- mean(abs((as.matrix(preds_ivae_filtered[step1_ahead_inds, 1]) - EEA_sub_test_filtered[step1_ahead_inds, "mean_O3"])), na.rm = TRUE)
ivae_rmse1 <- sqrt(mean((as.matrix(preds_ivae_filtered[step1_ahead_inds, 1]) - EEA_sub_test_filtered[step1_ahead_inds, "mean_O3"])^2, na.rm = TRUE))

step10_ahead_inds <- which(EEA_sub_test_filtered$time_numeric < (min(EEA_sub_test_filtered$time_numeric) + 10))
ivae_mae10 <- mean(abs((as.matrix(preds_ivae_filtered[step10_ahead_inds, 1]) - EEA_sub_test_filtered[step10_ahead_inds, "mean_O3"])), na.rm = TRUE)
ivae_rmse10 <- sqrt(mean((as.matrix(preds_ivae_filtered[step10_ahead_inds, 1]) - EEA_sub_test_filtered[step10_ahead_inds, "mean_O3"])^2, na.rm = TRUE))
print(paste0("iVAE MAE (1-step-ahead): ", ivae_mae1))
print(paste0("iVAE RMSE (1-step-ahead): ", ivae_rmse1))
print(paste0("iVAE MAE (10-steps-ahead): ", ivae_mae10))
print(paste0("iVAE RMSE (10-steps-ahead): ", ivae_rmse10))

bootstrap_results_1 <- bootstrap_errors(preds_ivae_filtered[step1_ahead_inds, 1], EEA_sub_test_filtered[step1_ahead_inds, "mean_O3"], n_bootstrap = 1000, seed = seed)
bootstrap_results_10 <- bootstrap_errors(preds_ivae_filtered[step10_ahead_inds, 1], EEA_sub_test_filtered[step10_ahead_inds, "mean_O3"], n_bootstrap = 1000, seed = seed)
print("Bootstrap results (1-step-ahead):")
print(bootstrap_results_1)
print("Bootstrap results (10-steps-ahead):")
print(bootstrap_results_10)


# Plotting


# FORECAST PLOTS:
min_time_test <- (min(EEA_sub_test_filtered$time_numeric))
step2_ahead_inds <- which(EEA_sub_test_filtered$time_numeric == (min_time_test + 1))
step3_ahead_inds <- which(EEA_sub_test_filtered$time_numeric == (min_time_test + 2))
step4_ahead_inds <- which(EEA_sub_test_filtered$time_numeric == (min_time_test + 3))
step5_ahead_inds <- which(EEA_sub_test_filtered$time_numeric == (min_time_test + 4))
step6_ahead_inds <- which(EEA_sub_test_filtered$time_numeric == (min_time_test + 5))
step7_ahead_inds <- which(EEA_sub_test_filtered$time_numeric == (min_time_test + 6))
step8_ahead_inds <- which(EEA_sub_test_filtered$time_numeric == (min_time_test + 7))
step9_ahead_inds <- which(EEA_sub_test_filtered$time_numeric == (min_time_test + 8))
step10_ahead_inds <- which(EEA_sub_test_filtered$time_numeric == (min_time_test + 9))

o3_pred_errors <- abs(preds_ivae_filtered[, 1] - EEA_sub_test_filtered$mean_O3)
step1preds_df <- as.data.frame(cbind(EEA_sub_test_filtered[step1_ahead_inds, ], preds_ivae_filtered[step1_ahead_inds, ], o3_pred_errors[step1_ahead_inds]))
step2preds_df <- as.data.frame(cbind(EEA_sub_test_filtered[step2_ahead_inds, ], preds_ivae_filtered[step2_ahead_inds, ], o3_pred_errors[step2_ahead_inds]))
step3preds_df <- as.data.frame(cbind(EEA_sub_test_filtered[step3_ahead_inds, ], preds_ivae_filtered[step3_ahead_inds, ], o3_pred_errors[step3_ahead_inds]))
step4preds_df <- as.data.frame(cbind(EEA_sub_test_filtered[step4_ahead_inds, ], preds_ivae_filtered[step4_ahead_inds, ], o3_pred_errors[step4_ahead_inds]))
step5preds_df <- as.data.frame(cbind(EEA_sub_test_filtered[step5_ahead_inds, ], preds_ivae_filtered[step5_ahead_inds, ], o3_pred_errors[step5_ahead_inds]))
step6preds_df <- as.data.frame(cbind(EEA_sub_test_filtered[step6_ahead_inds, ], preds_ivae_filtered[step6_ahead_inds, ], o3_pred_errors[step6_ahead_inds]))
step7preds_df <- as.data.frame(cbind(EEA_sub_test_filtered[step7_ahead_inds, ], preds_ivae_filtered[step7_ahead_inds, ], o3_pred_errors[step7_ahead_inds]))
step8preds_df <- as.data.frame(cbind(EEA_sub_test_filtered[step8_ahead_inds, ], preds_ivae_filtered[step8_ahead_inds, ], o3_pred_errors[step8_ahead_inds]))
step9preds_df <- as.data.frame(cbind(EEA_sub_test_filtered[step9_ahead_inds, ], preds_ivae_filtered[step9_ahead_inds, ], o3_pred_errors[step9_ahead_inds]))
step10preds_df <- as.data.frame(cbind(EEA_sub_test_filtered[step10_ahead_inds, ], preds_ivae_filtered[step10_ahead_inds, ], o3_pred_errors[step10_ahead_inds]))
names(step1preds_df)[ncol(step1preds_df)] <- "o3_pred_error"
names(step2preds_df)[ncol(step2preds_df)] <- "o3_pred_error"
names(step3preds_df)[ncol(step3preds_df)] <- "o3_pred_error"
names(step4preds_df)[ncol(step4preds_df)] <- "o3_pred_error"
names(step5preds_df)[ncol(step5preds_df)] <- "o3_pred_error"
names(step6preds_df)[ncol(step6preds_df)] <- "o3_pred_error"
names(step7preds_df)[ncol(step7preds_df)] <- "o3_pred_error"
names(step8preds_df)[ncol(step8preds_df)] <- "o3_pred_error"
names(step9preds_df)[ncol(step9preds_df)] <- "o3_pred_error"
names(step10preds_df)[ncol(step10preds_df)] <- "o3_pred_error"

all_preds <- c(
  step1preds_df$o3_pred_error, step2preds_df$o3_pred_error, step3preds_df$o3_pred_error, step4preds_df$o3_pred_error, step5preds_df$o3_pred_error, step6preds_df$o3_pred_error, step7preds_df$o3_pred_error, step8preds_df$o3_pred_error, step9preds_df$o3_pred_error, step10preds_df$o3_pred_error
)
color_min <- 0
color_max <- 42

mean(step1preds_df$o3_pred_error, na.rm = TRUE)
mean(step2preds_df$o3_pred_error, na.rm = TRUE)
mean(step3preds_df$o3_pred_error, na.rm = TRUE)
mean(step4preds_df$o3_pred_error, na.rm = TRUE)
mean(step5preds_df$o3_pred_error, na.rm = TRUE)
mean(step6preds_df$o3_pred_error, na.rm = TRUE)
mean(step7preds_df$o3_pred_error, na.rm = TRUE)
mean(step8preds_df$o3_pred_error, na.rm = TRUE)
mean(step9preds_df$o3_pred_error, na.rm = TRUE)
mean(step10preds_df$o3_pred_error, na.rm = TRUE)

library(ggplot2)
library(viridis)
error_plot_spatial <- function(preds_df, step_num, legend = FALSE) {
  preds_df$outlier_label <- ifelse(preds_df$o3_pred_error >= color_max, round(preds_df$o3_pred_error, 0), NA)
  g <- ggplot(preds_df, aes(x = X / 1000, y = Y / 1000, col = o3_pred_error)) +
    geom_point(data = subset(preds_df, !is.na(o3_pred_error) & o3_pred_error < color_max), aes(col = o3_pred_error)) + 
    geom_point(data = subset(preds_df, is.na(o3_pred_error)), color = "black", shape = 1) +
    geom_point(data = subset(preds_df, o3_pred_error >= color_max), color = "red", shape = 4) +
    geom_text(data = subset(preds_df, !is.na(outlier_label)), aes(label = outlier_label), vjust = -0.5, color = "red", size = 3, fontface = "bold") +
    scale_color_viridis(name = "Absolute error", limits = c(color_min, color_max)) +
    labs(title = paste0("Step ", step_num), x = "X (km)", y = "Y (km)") +
    theme_minimal() +
    coord_fixed()
    if (!legend) {
      g <- g + theme(legend.position = "none")
      return(g)
    }
    return(g)
}
step1 <- error_plot_spatial(step1preds_df, 1)
step2 <- error_plot_spatial(step2preds_df, 2)
step3 <- error_plot_spatial(step3preds_df, 3)
step4 <- error_plot_spatial(step4preds_df, 4)
step5 <- error_plot_spatial(step5preds_df, 5)
step6 <- error_plot_spatial(step6preds_df, 6)
step7 <- error_plot_spatial(step7preds_df, 7)
step8 <- error_plot_spatial(step8preds_df, 8)
step9 <- error_plot_spatial(step9preds_df, 9)
step10 <- error_plot_spatial(step10preds_df, 10, legend = TRUE)

step10_with_legend <- step10
step10 <- step10 + theme(legend.position = "none")
library(cowplot)
library(gridExtra)
legend <- get_legend(step10_with_legend)

plot_grid <- plot_grid(step1, step2, step3, step4, step5, step6, step7, step8, step9, step10, ncol = 3)

final_plot <- plot_grid(plot_grid, legend, rel_widths = c(3, 0.3))
final_plot
