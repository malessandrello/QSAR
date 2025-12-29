library(dplyr)
library(tidyverse)
library(caret)
library(FactoMineR)
library(ggbiplot)
library(xgboost)
library(randomForest)
set.seed(101)
data <- read_delim("qsar_fish_toxicity.csv", col_names = FALSE, delim = ";" ) %>% set_names(c("CIC0",
                                                                                              "SM1",
                                                                                              "GATS1i",
                                                                                              "NdsCH",
                                                                                              "NdssC",
                                                                                              "MLOGP",
                                                                                              "LC50"))


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


grid <- expand.grid(
  eta = c(0.01, 0.1, 0.3),
  max_depth = c(4, 6, 8),
  subsample = c(0.7, 0.8, 0.9),
  colsample_bytree = c(0.7, 0.8, 0.9)
)

param <- list(max_depth = 8,
               eta = 0.1,
                gamma = 0.8,
                min_child_weight = 15,
               subsample = 0.7,
                lambda = 5,
                alpha = 4,
               colsample_bytree = 0.8,
               eval_metric = "rmse")

dtrain <- as.matrix(train_x)
dtest <-  as.matrix(test_x)

dtrain <- xgb.DMatrix(dtrain, label = train_y)
dtest <- xgb.DMatrix(dtest, label = test_y)

watchlist <- list(train = dtrain, eval = dtest)

set.seed(101)
# results <- apply(grid, 1, function(params) {
#   xgb.cv(
#     params = as.list(params),
#     data = dtrain,
#     nrounds = 100,
#     nfold = 5,
#     early_stopping_rounds = 10,
#     verbose = TRUE
#   )
# })
set.seed(101)
bst <- xgb.train(param, dtrain, nrounds = 1000, watchlist, early_stopping_rounds = 10)


pred <- predict(bst, dtest)


MAE(pred, test_set$label)
RMSE(pred, test_set$label)
Metrics::mdae(pred, test_set$label)

plot(pred, test_set$label, abline(a = 0, b = 1))


rf <- randomForest::randomForest(label ~.,
                                 data = train_set,
                                  mtry = 2,
                                  ntree = 1000 
                           )

predict_rf <- predict(rf, test_set, type = "response")

MAE(predict_rf, test_set$label)
RMSE(predict_rf, test_set$label)
Metrics::mdae(predict_rf, test_set$label)

meta_data <- tibble(pred, predict_rf)
means <- rowMeans(meta_data)
RMSE(means, test_set$label)
Metrics::mdae(means, test_set$label)

plot(means, test_set$label, abline(a = 0, b = 1))

IQR(data$label)
quantile(data$label)
outlier <- 4.91 + 1.5*IQR(data$label)