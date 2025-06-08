rm(list = ls())
library(dplyr)
library(caret)
library(modelr)
library(glmnet)
library(grf)

setwd("C:/Users/vanal/OneDrive/Desktop/Uni Work/Machine Learning/Homework")

###Question 1
data <- read.csv("coursework.csv")
data <- data[complete.cases(data),]
data_reduced <- data[,c(1,2,3,12)]
data_reduced <- data_reduced[complete.cases(data_reduced),]
data_reduced <- data_reduced %>%
  mutate(postal_code = as.factor(postal_code)) %>%
  model.matrix(~postal_code + treat + math_outcome + reading_outcome + 0, data = .)
data_reduced <- as.data.frame(data_reduced)
treated_data <- filter(data_reduced, treat == 1)
untreated_data <- filter(data_reduced, treat == 0)

#####Question 2
set.seed(42)
trainIndex <- createDataPartition(data_reduced$treat, p = .8, list = FALSE)
data_train <- data_reduced[trainIndex,]
data_test <- data_reduced[-trainIndex,]

y_test_math <- data_test$math_outcome
y_test_reading <- data_test$reading_outcome

ols_math <- lm(math_outcome ~ treat, data = data_train)
ols_reading <- lm(reading_outcome ~ treat, data = data_train)

x_train <- data_train[,c(2:ncol(data_train))]
y_train_math <- data_train$math_outcome
y_train_reading <- data_train$reading_outcome

X_train_lasso <- model.matrix(~ . - math_outcome - reading_outcome, data_train)
X_test_lasso <- model.matrix(~ . - math_outcome - reading_outcome, data_test)
y_train_math_lasso <- data_train$math_outcome
y_train_reading_lasso <- data_train$reading_outcome

cv_lasso_math <- cv.glmnet(X_train_lasso, y_train_math_lasso, alpha = 1, nfolds = 5)
cv_lasso_reading <- cv.glmnet(X_train_lasso, y_train_reading_lasso, alpha = 1, nfolds = 5)

#####Question 3
pred_ols_math <- predict(ols_math, newdata = data_test)
pred_ols_reading <- predict(ols_reading, newdata = data_test)
  
# Predicting with LASSO
pred_lasso_math <- predict(cv_lasso_math, newx = X_test_lasso, s = "lambda.min")
pred_lasso_reading <- predict(cv_lasso_reading, newx = X_test_lasso, s = "lambda.min")
  
mse_ols_math <- mean((y_test_math - pred_ols_math)^2)
mse_ols_reading <- mean((y_test_reading - pred_ols_reading)^2)
mse_lasso_math <- mean((y_test_math - pred_lasso_math)^2)
mse_lasso_reading <- mean((y_test_reading - pred_lasso_reading)^2)

mse_ols_math
mse_ols_reading
mse_lasso_math
mse_lasso_reading

#######Question 4
X_test_treated <- X_test_lasso
X_test_untreated <- X_test_lasso
X_test_treated[, 'treat'] <- 1
X_test_untreated[, 'treat'] <- 0

pred_math_treated <- predict(cv_lasso_math, newx = X_test_treated, s = "lambda.min")
pred_reading_treated <- predict(cv_lasso_reading, newx = X_test_treated, s = "lambda.min")
pred_math_untreated <- predict(cv_lasso_math, newx = X_test_untreated, s = "lambda.min")
pred_reading_untreated <- predict(cv_lasso_reading, newx = X_test_untreated, s = "lambda.min")

ate_math <- mean(pred_math_treated - pred_math_untreated)
ate_reading <- mean(pred_reading_treated - pred_reading_untreated)
ate_math
ate_reading

#######Question 5
Y_train <- data_train$math_outcome
T_train <- data_train$treat
X_train <- data_train[, !(names(data_train) %in% c("math_outcome", "reading_outcome", "treat"))]

X_train <- model.matrix(~ . - 1, data = X_train)

causal_forest_model <- causal_forest(X_train, Y_train, T_train)
print(causal_forest_model)

X_test <- data_test[, !(names(data_test) %in% c("math_outcome", "reading_outcome", "treat"))]
X_test <- model.matrix(~ . - 1, data = X_test)

ite_pred <- predict(causal_forest_model, X_test)$predictions

head(ite_pred)
