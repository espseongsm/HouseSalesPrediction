rm (list=ls())
cat("\014")

library(dplyr)
# library(MASS)
library(glmnet)
library(randomForest)
library(ggplot2)
# library(reshape)
# library(gridExtra)
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


# test data
test = read.csv('Data Science ZExercise_TEST_CONFIDENTIAL2.csv', row.names = 'PropertyID')
test = subset(test, select = -c(censusblockgroup, Usecode, ViewType, 
                                ZoneCodeCounty, TransDate))
round(colSums(is.na(test))/dim(test)[1]*100,1)

# handle missing values of 4 predictors
train$GarageSquareFeet[is.na(train$GarageSquareFeet)] = 0
train$BGMedHomeValue[is.na(train$BGMedHomeValue)] = median(na.omit(train$BGMedHomeValue))
train$BGMedYearBuilt[is.na(train$BGMedYearBuilt)] = mean(na.omit(train$BGMedYearBuilt))
train$BGMedRent[is.na(train$BGMedRent)] = mean(na.omit(train$BGMedRent))

test$GarageSquareFeet[is.na(test$GarageSquareFeet)] = 0
test$BGMedHomeValue[is.na(test$BGMedHomeValue)] = median(na.omit(test$BGMedHomeValue))
test$BGMedYearBuilt[is.na(test$BGMedYearBuilt)] = mean(na.omit(test$BGMedYearBuilt))
test$SaleDollarCnt = 0
test$BGMedRent[is.na(test$BGMedRent)] = mean(na.omit(test$BGMedRent))

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
coef(lasso.mod)

# el-net
cv.elasticnet = cv.glmnet(x = X, y = y, alpha = 0.5, family = "gaussian", 
                          intercept = T, nfolds = 5, grouped = FALSE, , type.measure = "mae")
bestlam_el = cv.elasticnet$lambda.min
elasticnet.mod = glmnet(X, y, alpha = 0.5, family = "gaussian",
                        intercept = T, lambda = bestlam_el)
plot(cv.elasticnet)
coef(elasticnet.mod)

# ridge
cv.ridge = cv.glmnet(x = X, y = y, alpha = 0, family = "gaussian", 
                     intercept = T, nfolds = 5, grouped = FALSE, type.measure = "mae")
bestlam_ridge = cv.ridge$lambda.min
ridge.mod = glmnet(X, y, alpha = 0, family = "gaussian",
                   intercept = T, lambda = bestlam_ridge)

plot(cv.ridge)
coef(ridge.mod)

# test prediction
#rf
rf = randomForest(X, y, mtry = p/2, importance = T, ntree = 175)
rf_test_y_hat = predict(rf, newdata = test_X)
write.csv(cbind(rf_test_y_hat,test_X), 'test_prediction.csv')

xtable(round(cbind(coef(lasso.mod)[-1], coef(elasticnet.mod)[-1], coef(ridge.mod)[-1], rf$importance)[,(1:4)],3))
