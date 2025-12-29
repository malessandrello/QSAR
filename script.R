library(dplyr)
library(tidyverse)
library(caret)
library(FactoMineR)
library(ggbiplot)
library(xgboost)
set.seed(101)
data <- read_delim("qsar_fish_toxicity.csv", col_names = FALSE, delim = ";" ) %>% set_names(c("CIC0",
                                                       "SM1_Dz(Z)",
                                                       "GATS1i",
                                                       "NdsCH",
                                                       "NdssC",
                                                       "MLOGP",
                                                       "LC50"))
colSums(is.na(data))
sum(is.null(data[1:nrow(data),]) == TRUE)
sum(is.na(data[1:nrow(data),]) == TRUE)

# summary(data$LC50)
# 
# 
# data %>%
#   ggplot(aes(LC50)) +
#   geom_histogram()
# 
# data %>%
#   ggplot(aes(`SM1_Dz(Z)`, LC50))+
#   geom_point()
# 
# data %>%
#   ggplot(aes(GATS1i, MLOGP))+
#   geom_point()
# 
# data %>%
#   ggplot(aes(GATS1i, LC50))+
#   geom_point()
# 
# data %>%
#   ggplot(aes(NdsCH, LC50))+
#   geom_point()
# 
# data %>%
#   ggplot(aes(NdssC, LC50))+
#   geom_point()
# 
# data %>%
#   ggplot(aes(MLOGP, LC50))+
#   geom_point()

data <- data %>% 
  rename("label" = "LC50") %>% 
  filter(label < 7.54)

index <- createDataPartition(data$label, times = 1, p = 0.8, list = FALSE )

train_set <- data[index,]
train_x <- train_set %>% select(-label)
train_y <- train_set %>% select(label) %>% pull()

test_set <- data[-index,]
test_x <- test_set %>% select(-label)
test_y <- test_set %>% select(label) %>% pull()


# pca_train_x<- prcomp(train_x, scale. = TRUE)
# 
# train_x_pca <- pca_train_x$x[,1:4]
# 
# pca_test_x <- prcomp(test_x, scale. = TRUE)
# test_x_pca <- pca_test_x$x[,1:4]
# test_set_pca <- as_tibble(cbind(test_x_pca, label = test_set$label))


trControl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

tune_grid1 <- expand.grid(nrounds = c(100), max_depth = c(1), eta = c(0.1, 0.3, 0.5, 1,2),
                         gamma = c(0), colsample_bytree = c(1),
                         min_child_weight = c(1), subsample = c(1))



model_train1 <- caret::train(x = train_x,
                            y = train_y, 
                            method= "xgbTree",
                            trControl = trControl,
                            tuneGrid = tune_grid1)

predict1 <- predict(model_train1, test_set)

RMSE(predict1, test_set$label)
min(model_train1$results$RMSE)
model_train1$bestTune

set.seed(101)
tune_grid2 <- expand.grid(nrounds = c(100), max_depth = c(1), eta = c(0.1),
                          gamma = c(0), colsample_bytree = c(1),
                          min_child_weight = c(14), subsample = c(1))

model_train2 <- caret::train(x = train_x,
                             y = train_y, 
                             method= "xgbTree",
                             trControl = trControl,
                             tuneGrid = tune_grid2)


predict2 <- predict(model_train2, test_set)

RMSE(predict2, test_set$label)
min(model_train2$results$RMSE)
model_train2$bestTune

tune_grid3 <- expand.grid(nrounds = c(100), max_depth = c(1), eta = c(0.1),
                          gamma = c(0), colsample_bytree = c(0.5, 0.8, 1),
                          min_child_weight = c(14), subsample = c(0.5, 0.8, 1))

model_train3 <- caret::train(x = train_x,
                             y = train_y, 
                             method= "xgbTree",
                             trControl = trControl,
                             tuneGrid = tune_grid3)

predict3 <- predict(model_train3, test_set)

RMSE(predict3, test_set$label)
min(model_train3$results$RMSE)
model_train3$bestTune

tune_grid4 <- expand.grid(nrounds = c(100), max_depth = c(1), eta = c(0.1),
                          gamma = c(0,0.5,0.8,1,2), colsample_bytree = c(1),
                          min_child_weight = c(14), subsample = c(0.5))

model_train4 <- caret::train(x = train_x,
                             y = train_y, 
                             method= "xgbTree",
                             trControl = trControl,
                             tuneGrid = tune_grid4)

predict4 <- predict(model_train4, test_set)

RMSE(predict4, test_set$label)
min(model_train4$results$RMSE)
model_train4$bestTune

tune_grid5 <- expand.grid(nrounds = c(4, 10, 20, 30, 50, 100), max_depth = c(1), eta = c(0.1),
                          gamma = c(0.5), colsample_bytree = c(1),
                          min_child_weight = c(14), subsample = c(0.5))

model_train5 <- caret::train(x = train_x,
                             y = train_y, 
                             method= "xgbTree",
                             trControl = trControl,
                             tuneGrid = tune_grid5)

predict5 <- predict(model_train5, test_set)

RMSE(predict5, test_set$label)
min(model_train5$results$RMSE)
model_train5$bestTune



set.seed(101)

dtrain <- as.matrix(train_x)
dtest <-  as.matrix(test_x)

dtrain <- xgb.DMatrix(dtrain, label = train_y)
dtest <- xgb.DMatrix(dtest, label = test_y)

watchlist <- list(train = dtrain, eval = dtest)

param <- list(max_depth = 8, 
              eta = 0.1,
              gamma = 0.8,
              min_child_weight = 15,
              subsample = 0.7,
               lambda = 5,
               alpha = 4,
              colsample_bytree = 0.8,
              eval_metric = "rmse")

bst <- xgb.train(param, dtrain, nrounds = 1000, watchlist, early_stopping_rounds = 50)



pred <- predict(bst, dtest)


MAE(pred, test_set$label)
RMSE(pred, test_set$label)
Metrics::mdae(pred, test_set$label)

plot(pred, test_set$label, abline(a = 0, b = 1))

# 
# 
# pca_train_x<- prcomp(train_x, scale. = TRUE)
# 
# train_x_pca <- pca_train_x$x[,1:3]
# 
# pca_test_x <- prcomp(test_x, scale. = TRUE)
# test_x_pca <- pca_test_x$x[,1:3]
# 
# dtrain_pca <- as.matrix(train_x_pca)
# dtest_pca <-  as.matrix(test_x_pca)
# 
# dtrain <- xgb.DMatrix(dtrain_pca, label = train_y)
# dtest <- xgb.DMatrix(dtest_pca, label = test_y)
# 
# watchlist <- list(train = dtrain, eval = dtest)
# 
# param <- list(max_depth = 6, eta = 0.5,
#               gamma = 1,
#               min_child_weight = 10,
#               subsample = 1,
#               lambda = 5,
#               alpha = 4,
#               colsample_bytree = 1,
#               eval_metric = "rmse")
# 
# bst <- xgb.train(param, dtrain, nrounds = 100, watchlist, early_stopping_rounds = 10)
# 
# 
# 
# pred <- predict(bst, dtest)
# 
# 
# MAE(pred, test_set$label)
# RMSE(pred, test_set$label)
# 
# plot(pred, test_set$label, abline(a = 0, b = 1))
# 
# 
# model_train7 <- caret::train(label ~ ., 
#                              data = train_set,
#                              method= "rf")
# 
# pred1 <- predict(model_train7, test_set)
# 
# RMSE(pred1, test_set$label)
# MAE(pred1, test_set$label)
# 
# min(model_train7$results$RMSE)
# model_train7$bestTune

