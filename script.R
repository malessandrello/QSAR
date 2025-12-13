library(dplyr)
library(tidyverse)
library(caret)
library(xgboost)
set.seed(101)
data <- read_delim("qsar_fish_toxicity.csv", col_names = FALSE, delim = ";" ) %>% set_names(c("CIC0",
                                                       "SM1_Dz(Z)",
                                                       "GATS1i",
                                                       "NdsCH",
                                                       "NdssC",
                                                       "MLOGP",
                                                       "LC50"))

# sum(is.null(data[1:nrow(data),]) == TRUE)
# sum(is.na(data[1:nrow(data),]) == TRUE)
# 
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
  rename("label" = "LC50")

index <- createDataPartition(data$label, times = 1, p = 0.8, list = FALSE )

train_set <- data[index,]
train_x <- train_set %>% select(-label)
train_y <- train_set %>% select(label) %>% pull()

test_set <- data[-index,]

trControl <- trainControl(method = "cv", number = 5)

tune_grid1 <- expand.grid(nrounds = c(4), max_depth = c(1,2,3,4,5, 6, 10), eta = c(0.3),
                         gamma = c(0), colsample_bytree = c(1),
                         min_child_weight = c( 5,6,10, 20), subsample = c(1))



model_train1 <- caret::train(x = train_x,
                            y = train_y, 
                            method= "xgbTree",
                            metric = "RMSE",
                            trControl = trControl,
                            tuneGrid = tune_grid1)

predict1 <- predict(model_train1, test_set)

RMSE1 <- RMSE(predict1, test_set$label)

tune_grid2 <- expand.grid(nrounds = c(10), max_depth = c(6), eta = c(0.3),
                          gamma = c(0), colsample_bytree = c(0.1, 0.5, 0.7),
                          min_child_weight = c(6), subsample = c(0.1, 0.5, 0.7))

model_train2 <- caret::train(x = train_x,
                             y = train_y, 
                             method= "xgbTree",
                             metric = "RMSE",
                             trControl = trControl,
                             tuneGrid = tune_grid2)


predict2 <- predict(model_train2, test_set)

RMSE2 <- RMSE(predict2, test_set$label)

tune_grid3 <- expand.grid(nrounds = c(10), max_depth = c(6), eta = c(0.3),
                          gamma = c(0.5, 1, 2, 5), colsample_bytree = c(0.7),
                          min_child_weight = c(6), subsample = c(0.5))

model_train3 <- caret::train(x = train_x,
                             y = train_y, 
                             method= "xgbTree",
                             metric = "RMSE",
                             trControl = trControl,
                             tuneGrid = tune_grid3)

predict3 <- predict(model_train3, test_set)

RMSE3 <- RMSE(predict3, test_set$label)

tune_grid4 <- expand.grid(nrounds = c(100), max_depth = c(3), eta = c(0.01, 0.1,0.5, 1),
                          gamma = c(1), colsample_bytree = c(0.7),
                          min_child_weight = c(10), subsample = c(0.7))

model_train4 <- caret::train(x = train_x,
                             y = train_y, 
                             method= "xgbTree",
                             metric = "RMSE",
                             trControl = trControl,
                             tuneGrid = tune_grid4)

predict4 <- predict(model_train4, test_set)

RMSE4 <- RMSE(predict4, test_set$label)

tune_grid5 <- expand.grid(nrounds = c(10, 20, 30, 50, 100), max_depth = c(3), eta = c(0.1),
                          gamma = c(1), colsample_bytree = c(0.7),
                          min_child_weight = c(10), subsample = c(0.7))

model_train5 <- caret::train(x = train_x,
                             y = train_y, 
                             method= "xgbTree",
                             metric = "RMSE",
                             trControl = trControl,
                             tuneGrid = tune_grid5)

predict5 <- predict(model_train5, test_set)

RMSE5 <- RMSE(predict5, test_set$label)

tune_grid6 <- expand.grid(nrounds = c(30), max_depth = c(1), eta = c(0.1),
                          gamma = c(1), colsample_bytree = c(0.7),
                          min_child_weight = c(10), subsample = c(0.5))

model_train6 <- caret::train(x = train_x,
                             y = train_y, 
                             method= "xgbTree",
                             metric = "RMSE",
                             trControl = trControl,
                             tuneGrid = tune_grid6)

predict6 <- predict(model_train6, test_set)

RMSE6 <- RMSE(predict6, test_set$label)



test_set2_index <- createDataPartition(test_set$label, times = 1, p = 0.8, list = FALSE )
test_set2 <- test_set[test_set2_index,]
test_set3 <- test_set[-test_set2_index,]


dtrain <- as.matrix(train_x)
dtest <-  as.matrix(test_set2)

dtrain <- xgb.DMatrix(dtrain, label = train_set$label)
dtest <- xgb.DMatrix(dtest, label = test_set2$label)

watchlist <- list(train = dtrain, eval = dtest)

param <- list(max_depth = 2, eta = 1, objective = "reg:squarederror",
              eval_metric = "rmse")

bst <- xgb.train(param, dtrain, nrounds = 4, watchlist)

dtest3 <- xgb.DMatrix(as.matrix(test_set3))

pred <- predict(bst, dtest3)


RMSE(pred, test_set3$label)



