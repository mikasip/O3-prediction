spatiotemporal_idw <- function(data, coords_time, coords_time_pred, 
                              spatial_power = 2, temporal_power = 1,
                              max_dist = Inf, max_time_diff = Inf) {
  
  predictions <- numeric(nrow(coords_time_pred))

  train_coords <- coords_time[, 1:2]
  pred_coords <- coords_time_pred[, 1:2]
  unique_spatial_train <- unique(train_coords)
  unique_spatial_pred <- unique(pred_coords)
  unique_times_train <- unique(coords_time[, 3])
  unique_times_pred <- unique(coords_time_pred[, 3])
  spatial_dist_matrix <- rdist::cdist(unique_spatial_pred, unique_spatial_train)
  temporal_dist_matrix <- abs(outer(unique_times_pred, unique_times_train, "-"))
  
  # Pre-compute weight matrices
  spatial_weights_matrix <- 1 / (spatial_dist_matrix^spatial_power + 1e-10)
  temporal_weights_matrix <- 1 / (temporal_dist_matrix^temporal_power + 1e-10)
  
  # Apply constraints
  spatial_weights_matrix[spatial_dist_matrix > max_dist] <- 0
  temporal_weights_matrix[temporal_dist_matrix > max_time_diff] <- 0
  
  # Create lookup indices
  spatial_train_idx <- match(
    paste(train_coords[, 1], train_coords[, 2]), 
    paste(unique_spatial_train[, 1], unique_spatial_train[, 2])
  )
  spatial_pred_idx <- match(
    paste(pred_coords[, 1], pred_coords[, 2]), 
    paste(unique_spatial_pred[, 1], unique_spatial_pred[, 2])
  )
  temporal_train_idx <- match(
    coords_time[, 3], unique_times_train
  )
  temporal_pred_idx <- match(
    coords_time_pred[, 3], unique_times_pred
  )
    
  # Calculate predictions
  predictions <- numeric(nrow(coords_time_pred))
  
  for (i in 1:nrow(coords_time_pred)) {
    # Get relevant weight slices
    spatial_weights <- spatial_weights_matrix[spatial_pred_idx[i], spatial_train_idx]
    temporal_weights <- temporal_weights_matrix[temporal_pred_idx[i], temporal_train_idx]
    
    # Combined weights
    combined_weights <- spatial_weights * temporal_weights
    
    if (sum(combined_weights) == 0) {
      predictions[i] <- NA
    } else {
      combined_weights <- combined_weights / sum(combined_weights)
      predictions[i] <- sum(combined_weights * data)
    }
  }

  return(predictions)
}