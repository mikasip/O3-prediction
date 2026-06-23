library(ECoST)
library(tensorflow)
library(NonlinearBSS)
library(gstat)
library(sp)
library(spacetime)
source("helpers.R")

load("data/EEA_sub_val.RData")
load("data/EEA_sub_test.RData")
load("data/EEA_sub_train_val_aux_interpolated_ivae.RData")
EEA_sub_aux_interpolated <- EEA_sub_aux_train_interpolated_ivae

utm_test_coords <- to_utm(EEA_sub_test$Longitude, EEA_sub_test$Latitude)
EEA_sub_test$X <- utm_test_coords[,1]
EEA_sub_test$Y <- utm_test_coords[,2]

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
seed <- 29092025
t_fit_start <- proc.time()
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
t_fit_end <- proc.time()
print(paste0("HISTGNN fitting time (seconds): ", round(t_fit_end["elapsed"] - t_fit_start["elapsed"], 2)))

na_station_inds_test <- which(EEA_sub_test$AirQualityStation %in% na_stations)
test_data10 <- EEA_sub_test[-na_station_inds_test, ]
test_data10 <- test_data10[test_data10$time_numeric < (min(test_data10$time_numeric) + 10), ]
coords_time_test <- as.matrix(test_data10[, c("X", "Y", "time_numeric")])
new_order <- order(coords_time_test[, 1], coords_time_test[, 2], coords_time_test[, 3])
rev_new_order <- order(new_order)

t_inference_start <- proc.time()
preds <- predict(model)
t_inference_end <- proc.time()
print(paste0("HISTGNN inference time (seconds): ", round(t_inference_end["elapsed"] - t_inference_start["elapsed"], 2)))

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
bootstrap_results_1 <- bootstrap_errors(preds_ordered[seq(1, nrow(preds_ordered), by = 10), 1], test_data10[inds2, "mean_O3"], n_bootstrap = 1000, seed = seed)
bootstrap_results_10 <- bootstrap_errors(preds_ordered[, 1], test_data10[, "mean_O3"], n_bootstrap = 1000, seed = seed)
print("Bootstrap results (HISTGNN 1-step-ahead):")
print(bootstrap_results_1)
print("Bootstrap results (HISTGNN 10-steps-ahead):")
print(bootstrap_results_10)


load("../Italy_project/histgnn_preds_test.RData")
preds_ordered <- preds$predictions[rev_new_order, ]
colMeans(abs(preds_ordered[, 1:3] - test_data10[, c("mean_O3", "mean_NO2", "mean_PM10")]), na.rm = TRUE)
inds2 <- which(coords_time_test[,3] < (min(coords_time_test[,3]) + 1))
histgnn_mae1 <- mean(abs(preds_ordered[seq(1, nrow(preds_ordered), by = 10), 1] - test_data10[inds2, "mean_O3"]), na.rm = TRUE)


# FORECAST PLOTS:
min_time_test <- min(coords_time_test[, 3])
step1_ahead_inds <- which(coords_time_test[, 3] == min_time_test)
step2_ahead_inds <- which(coords_time_test[, 3] == (min_time_test + 1))
step3_ahead_inds <- which(coords_time_test[, 3] == (min_time_test + 2))
step4_ahead_inds <- which(coords_time_test[, 3] == (min_time_test + 3))
step5_ahead_inds <- which(coords_time_test[, 3] == (min_time_test + 4))
step6_ahead_inds <- which(coords_time_test[, 3] == (min_time_test + 5))
step7_ahead_inds <- which(coords_time_test[, 3] == (min_time_test + 6))
step8_ahead_inds <- which(coords_time_test[, 3] == (min_time_test + 7))
step9_ahead_inds <- which(coords_time_test[, 3] == (min_time_test + 8))
step10_ahead_inds <- which(coords_time_test[, 3] == (min_time_test + 9))

o3_pred_errors <- abs(preds_ordered[, 1] - test_data10[, "mean_O3"])

step1preds_df <- as.data.frame(cbind(test_data10[step1_ahead_inds, ], preds_ordered[step1_ahead_inds, ], o3_pred_error = o3_pred_errors[step1_ahead_inds]))
step2preds_df <- as.data.frame(cbind(test_data10[step2_ahead_inds, ], preds_ordered[step2_ahead_inds, ], o3_pred_error = o3_pred_errors[step2_ahead_inds]))
step3preds_df <- as.data.frame(cbind(test_data10[step3_ahead_inds, ], preds_ordered[step3_ahead_inds, ], o3_pred_error = o3_pred_errors[step3_ahead_inds]))
step4preds_df <- as.data.frame(cbind(test_data10[step4_ahead_inds, ], preds_ordered[step4_ahead_inds, ], o3_pred_error = o3_pred_errors[step4_ahead_inds]))
step5preds_df <- as.data.frame(cbind(test_data10[step5_ahead_inds, ], preds_ordered[step5_ahead_inds, ], o3_pred_error = o3_pred_errors[step5_ahead_inds]))
step6preds_df <- as.data.frame(cbind(test_data10[step6_ahead_inds, ], preds_ordered[step6_ahead_inds, ], o3_pred_error = o3_pred_errors[step6_ahead_inds]))
step7preds_df <- as.data.frame(cbind(test_data10[step7_ahead_inds, ], preds_ordered[step7_ahead_inds, ], o3_pred_error = o3_pred_errors[step7_ahead_inds]))
step8preds_df <- as.data.frame(cbind(test_data10[step8_ahead_inds, ], preds_ordered[step8_ahead_inds, ], o3_pred_error = o3_pred_errors[step8_ahead_inds]))
step9preds_df <- as.data.frame(cbind(test_data10[step9_ahead_inds, ], preds_ordered[step9_ahead_inds, ], o3_pred_error = o3_pred_errors[step9_ahead_inds]))
step10preds_df <- as.data.frame(cbind(test_data10[step10_ahead_inds, ], preds_ordered[step10_ahead_inds, ], o3_pred_error = o3_pred_errors[step10_ahead_inds]))

mean(o3_pred_errors[step1_ahead_inds], na.rm = TRUE)
mean(o3_pred_errors[step2_ahead_inds], na.rm = TRUE)
mean(o3_pred_errors[step3_ahead_inds], na.rm = TRUE)
mean(o3_pred_errors[step4_ahead_inds], na.rm = TRUE)
mean(o3_pred_errors[step5_ahead_inds], na.rm = TRUE)
mean(o3_pred_errors[step6_ahead_inds], na.rm = TRUE)
mean(o3_pred_errors[step7_ahead_inds], na.rm = TRUE)
mean(o3_pred_errors[step8_ahead_inds], na.rm = TRUE)
mean(o3_pred_errors[step9_ahead_inds], na.rm = TRUE)
mean(o3_pred_errors[step10_ahead_inds], na.rm = TRUE)

all_preds <- c(
  step1preds_df$o3_pred_error, step2preds_df$o3_pred_error, step3preds_df$o3_pred_error, step4preds_df$o3_pred_error, step5preds_df$o3_pred_error, step6preds_df$o3_pred_error, step7preds_df$o3_pred_error, step8preds_df$o3_pred_error, step9preds_df$o3_pred_error, step10preds_df$o3_pred_error
)
color_min <- 0
color_max <- 42 #max(color_max_histgnn, color_max_ivaear)

library(ggplot2)
library(viridis)
# Create function to generate plots for each step
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

