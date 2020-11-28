rm (list=ls())
cat("\014")

library(dplyr)
library(MASS)
library(glmnet)
library(randomForest)
library(ggplot2)
library(reshape)
library(gridExtra)
library(olsrr)
library(tidyverse)
library(xtable)

setwd("~/Documents/Zillow_ML_Intern")

# train data
train = read.csv("Data Science ZExercise_TRAINING_CONFIDENTIAL1.csv", row.names = 'PropertyID')
train = subset(train, select = -c(censusblockgroup, Usecode, ViewType, 
                                  ZoneCodeCounty, TransDate))
round(colSums(is.na(train) )/dim(train)[1]*100,1)
summary(train)
ggplot(train, aes(x=SaleDollarCnt)) + geom_histogram(color="black", fill="white", bins = 100)
ggplot(train, aes(x=BGMedHomeValue)) + geom_histogram(color="black", fill="white", bins = 100)
mean(train$SaleDollarCnt)
sd(train$SaleDollarCnt)
0.5e+6 == 500000
hist(train$Latitude)
hist(train$BGMedHomeValue)
hist(test$BGMedHomeValue)
hist(train$BGMedRent)
hist(train$BGMedYearBuilt)
cor(train$BGPctKids, train$BGMedAge)
hist(train$BuiltYear)
plot(train$BuiltYear ~ train$BGMedYearBuilt)
typeof(train$TransDate)
train$TransDate
barplot(train$ZoneCodeCounty)
sd(train$Latitude)/mean(train$Latitude)
plot(train$ZoneCodeCounty ~ train$Latitude)
sd(train$Longitude)
table(train$ZoneCodeCounty)

# test data
test = read.csv('Data Science ZExercise_TEST_CONFIDENTIAL2.csv', row.names = 'PropertyID')
test = subset(test, select = -c(censusblockgroup, Usecode, ViewType, 
                                ZoneCodeCounty, TransDate))
round(colSums(is.na(test))/dim(test)[1]*100,1)

# handle missing values of 4 predictors
train$GarageSquareFeet[is.na(train$GarageSquareFeet)] = 0
train$BGMedHomeValue[is.na(train$BGMedHomeValue)] = median(na.omit(train$BGMedHomeValue))
train$BGMedYearBuilt[is.na(train$BGMedYearBuilt)] = mean(na.omit(train$BGMedYearBuilt))

train_non_missing = subset(train, select = -c(SaleDollarCnt))[!is.na(train$BGMedRent),]
train_missing = subset(train, select = -c(SaleDollarCnt))[is.na(train$BGMedRent),]
train_missing_mod = lm(BGMedRent ~., train_non_missing)
summary(train_missing_mod)
ols_vif_tol(train_missing_mod)
train_missing_BGMedRent_estimate = predict(train_missing_mod, 
                                           train_missing[, -which(names(train_missing) %in% 
                                                                    c('BGMedRent'))])
train$BGMedRent[is.na(train$BGMedRent)] = train_missing_BGMedRent_estimate
mean(na.omit(train$BGMedRent))

test$GarageSquareFeet[is.na(test$GarageSquareFeet)] = 0
test$BGMedHomeValue[is.na(test$BGMedHomeValue)] = median(na.omit(test$BGMedHomeValue))
test$BGMedYearBuilt[is.na(test$BGMedYearBuilt)] = mean(na.omit(test$BGMedYearBuilt))
test$SaleDollarCnt = 0

test_non_missing = subset(test, select = -c(SaleDollarCnt))[!is.na(test$BGMedRent),]
test_missing = subset(test, select = -c(SaleDollarCnt))[is.na(test$BGMedRent),]
test_missing_mod = lm(BGMedRent ~., test_non_missing)
test_missing_BGMedRent_estimate = predict(test_missing_mod, 
                                          test_missing[, -which(names(test_missing) %in% 
                                                                    c('BGMedRent'))])
test$BGMedRent[is.na(test$BGMedRent)] = test_missing_BGMedRent_estimate

# convert character to date
train$TransDate = as.Date(train$TransDate, "%m/%d/%Y")
test$TransDate = as.Date(test$TransDate, "%m/%d/%Y")

train_and_test = rbind(train, test)

# one hot encoding
X = model.matrix(SaleDollarCnt~., train_and_test)[(1:11588),-1]
y = train$SaleDollarCnt
train_one_hot_encoded = data.frame(cbind(y, X))
p = dim(X)[2]
test_X = model.matrix(SaleDollarCnt~., train_and_test)[11589:15990,-1]
test_y = test$SaleDollarCnt
test_one_hot_encoded = data.frame(cbind(test_y, test_X))

# lasso
cv.lasso = cv.glmnet(x = X, y = y, alpha = 1, family = "gaussian", 
                     intercept = T, nfolds = 5, grouped = FALSE, type.measure = "mae")
bestlam_lasso = cv.lasso$lambda.min
lasso.mod = glmnet(X, y, alpha = 1, family = "gaussian",
                   intercept = T, lambda = bestlam_lasso)
plot(cv.lasso)
lasso_train_y_hat = predict(lasso.mod, X)
lasso_aape = mean(abs(lasso_train_y_hat - y)/y)
lasso_mape = median(abs(lasso_train_y_hat - y)/y)
coef(lasso.mod)
# el-net
cv.elasticnet = cv.glmnet(x = X, y = y, alpha = 0.5, family = "gaussian", 
                          intercept = T, nfolds = 5, grouped = FALSE, , type.measure = "mae")
bestlam_el = cv.elasticnet$lambda.min
elasticnet.mod = glmnet(X, y, alpha = 0.5, family = "gaussian",
                        intercept = T, lambda = bestlam_el)
el_train_y_hat = predict(elasticnet.mod, X)
el_aape = mean(abs(el_train_y_hat - y)/y)
el_mape = median(abs(el_train_y_hat - y)/y)
plot(cv.elasticnet)
coef(elasticnet.mod)
# ridge
cv.ridge = cv.glmnet(x = X, y = y, alpha = 0, family = "gaussian", 
                     intercept = T, nfolds = 5, grouped = FALSE, type.measure = "mae")
bestlam_ridge = cv.ridge$lambda.min
ridge.mod = glmnet(X, y, alpha = 0, family = "gaussian",
                   intercept = T, lambda = bestlam_ridge)
ridge_train_y_hat = predict(ridge.mod, X)
ridge_aape = mean(abs(ridge_train_y_hat - y)/y)
ridge_mape = median(abs(ridge_train_y_hat - y)/y)
plot(cv.ridge)
coef(ridge.mod)

# multiple regression
multiple_reg = lm(y ~., train_one_hot_encoded)
sum_multi = summary(multiple_reg)
multiple_reg_train_y_hat = predict(multiple_reg, data.frame(X))
multiple_reg_aape = mean(abs(multiple_reg_train_y_hat - y)/y)
vif_multi_reg = ols_vif_tol(multiple_reg)

# multiple regression with hybrid step wise feature selection
hybrid_selection = ols_step_both_p(multiple_reg)
plot(hybrid_selection)

# multiple regression with forward step wise feature selection
forward_selection = ols_step_forward_p(multiple_reg)
plot(forward_selection)

# multiple regression with backward step wise feature selection
backward_selection = ols_step_backward_p(multiple_reg)
plot(backward_selection)

# cross-validation for hyperparameter(no_of_trees) tuning
k = 5
indices = sample(1:nrow(train_one_hot_encoded))
folds = cut(indices, breaks = k, labels = FALSE)

no_of_trees = seq(25,300,25)
rf_cv_tree_AAPE = matrix(0, nrow = length(no_of_trees), ncol = k)

for (j in no_of_trees) {
  cat("testing no. of trees = ", j, "\n")
  for (i in 1:k) {
    cat("processing cross-validation fold #", i, "\n")
    
    val_indices = which(folds == i, arr.ind = TRUE) 
    val_X = X[val_indices,]
    val_y = y[val_indices]
    
    partial_X = X[-val_indices,] 
    partial_y = y[-val_indices]
    
    rf = randomForest(partial_X, partial_y, mtry = sqrt(p), importance = T, ntree = j)
    rf_val_y_hat = predict(rf, newdata = val_X)
    rf_aape = mean(abs(rf_val_y_hat - val_y)/val_y)
    
    rf_cv_tree_AAPE[(j/25), i] = rf_aape
  }
}

apply(rf_cv_tree_AAPE, 1, mean)
rf_cv_tree = read.csv('rf_cv_aape_trees.csv')
ggplot(data=rf_cv_tree, aes(x=trees, y=AAPE, group=1)) +
  geom_line(color="red")+
  geom_point()

# cross-validation for hyperparameter(no_of_predictors) tuning
rf_cv_predictor_AAPE = matrix(0, nrow = p, ncol = k)

for (j in 1:p) {
  cat("testing no. of trees = ", j, "\n")
  for (i in 1:k) {
    cat("processing cross-validation fold #", i, "\n")
    
    val_indices = which(folds == i, arr.ind = TRUE) 
    val_X = X[val_indices,]
    val_y = y[val_indices]
    
    partial_X = X[-val_indices,] 
    partial_y = y[-val_indices]
    
    rf = randomForest(partial_X, partial_y, mtry = j, importance = T, ntree = 175)
    rf_val_y_hat = predict(rf, newdata = val_X)
    rf_aape = mean(abs(rf_val_y_hat - val_y)/val_y)
    
    rf_cv_predictor_AAPE[j, i] = rf_aape
  }
}

apply(rf_cv_predictor_AAPE, 1, mean)
rf_cv_predictor = read.csv('rf_cv_aape_predictors.csv')
ggplot(data=rf_cv_predictor, aes(x=predictors, y=AAPE, group=1)) +
  geom_line(color="red")+
  geom_point()

# cross-validation for performance evaluation
k = 5
indices = sample(1:nrow(train_one_hot_encoded))
folds = cut(indices, breaks = k, labels = FALSE)

cv_aape = matrix(0, nrow = 5, ncol = 5)
cv_mape = matrix(0, nrow = 5, ncol = 5)

  for (i in 1:k) {
    cat("processing cross-validation fold #", i, "\n")
    
    val_indices = which(folds == i, arr.ind = TRUE) 
    val_X = X[val_indices,]
    val_y = y[val_indices]
    
    partial_X = X[-val_indices,] 
    partial_y = y[-val_indices]
    
    # lasso
    lasso.mod = glmnet(partial_X, partial_y, alpha = 1, family = "gaussian",
                       intercept = T, lambda = bestlam_lasso)
    lasso_val_y_hat = predict(lasso.mod, val_X)
    lasso_aape = mean(abs(lasso_val_y_hat - val_y)/val_y)
    lasso_mape = median(abs(lasso_val_y_hat - val_y)/val_y)
    cv_aape[1,i] = lasso_aape
    cv_mape[1,i] = lasso_mape
    
    # el-net
    elasticnet.mod = glmnet(partial_X, partial_y, alpha = 0.5, family = "gaussian",
                            intercept = T, lambda = bestlam_el)
    el_val_y_hat = predict(elasticnet.mod, val_X)
    el_aape = mean(abs(el_val_y_hat - val_y)/val_y)
    el_mape = median(abs(el_val_y_hat - val_y)/val_y)
    cv_aape[2,i] = el_aape
    cv_mape[2,i] = el_mape
    
    # ridge
    ridge.mod = glmnet(partial_X, partial_y, alpha = 0, family = "gaussian",
                       intercept = T, lambda = bestlam_ridge)
    ridge_val_y_hat = predict(ridge.mod, val_X)
    ridge_aape = mean(abs(ridge_val_y_hat - val_y)/val_y)
    ridge_mape = median(abs(ridge_val_y_hat - val_y)/val_y)
    cv_aape[3,i] = ridge_aape
    cv_mape[3,i] = ridge_mape
    
    # multiple regression
    multiple_reg = lm(y ~., train_one_hot_encoded[-val_indices, ])
    multiple_reg_val_y_hat = predict(multiple_reg, data.frame(val_X))
    multiple_reg_aape = mean(abs(multiple_reg_val_y_hat - val_y)/val_y)
    multiple_reg_mape = median(abs(multiple_reg_val_y_hat - val_y)/val_y)
    cv_aape[4,i] = multiple_reg_aape
    cv_mape[4,i] = multiple_reg_mape
    
    #rf
    rf = randomForest(partial_X, partial_y, mtry = p/2, importance = T, ntree = 175)
    rf_val_y_hat = predict(rf, newdata = val_X)
    rf_aape = mean(abs(rf_val_y_hat - val_y)/val_y)
    rf_mape = median(abs(rf_val_y_hat - val_y)/val_y)
    cv_aape[5,i] = rf_aape
    cv_mape[5,i] = rf_mape
  }
round(apply(cv_aape, 1, mean), 3)
round(apply(cv_mape, 1, median), 3)

xtable(round(cbind(coef(lasso.mod)[-1], coef(elasticnet.mod)[-1], coef(ridge.mod)[-1], rf$importance)[,(1:4)],3))
