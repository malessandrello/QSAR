library(dplyr)
library(tidyverse)
library(caret)
library(xgboost)

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
test_set <- data[-index,]

test_set2_index <- createDataPartition(test_set$label, times = 1, p = 0.8, list = FALSE )
test_set2 <- test_set[test_set2_index,]
test_set3 <- test_set[-test_set2_index,]


dtrain <- as.matrix(train_set)
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