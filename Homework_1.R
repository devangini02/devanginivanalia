rm(list = ls())
library(glmnet)
library(dplyr)
library(caret)

setwd("C:/Users/vanal/OneDrive/Desktop/Uni Work/Machine Learning/Homework")

data <- read.csv("HW1_data.csv")

data <- data.frame(data)


# Splitting the data into training and test sets
set.seed(123) # for reproducibility


outcomes <- data[,1]
features <- data[,2]

folds <- createFolds(outcomes, k = 5, list = TRUE, returnTrain = FALSE)

#generate technical regressors
feature_matrices <- lapply(1:20, function(k) {
  as.matrix(sapply(1:k, function(i) features^i))
})

#penalty grid for penalty values
penalty_grid <- seq(from = 0.01, to = 1, by = 0.01)


#to store prediction errors
prediction_errors <- list()

for (j in 1:length(folds)) {
  for (k in 1:20) {
    for (lambda in penalty_grid) {
      # Define the training and test sets
      train_set <- feature_matrices[[k]][-folds[[j]], ]
      test_set <- feature_matrices[[k]][folds[[j]], ]
      train_outcomes <- outcomes[-folds[[j]]]
      test_outcomes <- outcomes[folds[[j]]]
    }
  }
}

# Fit models and calculate prediction errors
# OLS
model_ols <- lm(train_outcomes ~ ., data = as.data.frame(train_set))
pred_ols <- predict(model_ols, newdata = as.data.frame(test_set))
error_ols <- sum((test_outcomes - pred_ols)^2)

# Lasso
model_lasso <- glmnet(train_set, train_outcomes, alpha = 1, lambda = lambda)
pred_lasso <- predict(model_lasso, s = lambda, newx = test_set)
error_lasso <- sum((test_outcomes - pred_lasso)^2)

# Ridge
model_ridge <- glmnet(train_set, train_outcomes, alpha = 0, lambda = lambda)
pred_ridge <- predict(model_ridge, s = lambda, newx = test_set)
error_ridge <- sum((test_outcomes - pred_ridge)^2)

# Store errors
prediction_errors[[paste(j, k, lambda)]] <- c(ols = error_ols, lasso = error_lasso, ridge = error_ridge)


#sum of squared errors
total_errors <- Reduce('+', prediction_errors)

#find optimal k and lambda 
best_combination <- sapply(c("ols", "lasso", "ridge"), function(model) {
  which.min(sapply(prediction_errors, function(x) x[model]))
})

min_error_lasso <- which.min(sapply(prediction_errors, function(x) x["lasso"]))
optimal_combination_lasso <- names(prediction_errors)[min_error_lasso]
optimal_k_lasso <- as.integer(strsplit(optimal_combination_lasso, " ")[[1]][2])
optimal_lambda_index <- as.integer(strsplit(optimal_combination_lasso, " ")[[1]][3])
optimal_lambda_lasso <- penalty_grid[optimal_lambda_index]

expanded_features <- sapply(1:optimal_k_lasso, function(k) features^k)

final_model_lasso <- glmnet(as.matrix(expanded_features), outcomes, alpha = 1, lambda = optimal_lambda_lasso)
plot(final_model_lasso)
