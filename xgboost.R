IBM_financial <- read.csv('~/Desktop/Machine Learning/Prophet Project/IBMFinancialStatements.csv')
data <- read.csv('~/Desktop/Machine Learning/Prophet Project/GoldmineIBM.csv')
IBM_volatility <- read.csv('~/Desktop/Machine Learning/Prophet Project/VolatilityIBM.csv')
 
library(dplyr)

data$DATE <- as.Date(data$DATE)
data <- data[, -which(sapply(data, class) == "character")]


# generator <- function(data, lookback, delay, min_index, max_index,
#                       shuffle = FALSE, batch_size = 20, step = 1, predseries) {
#   if (is.null(max_index)) max_index <- nrow(data) - delay - 1
#   i <- min_index + lookback
#   function() {
#     if (shuffle) {
#       rows <- sample(c((min_index+lookback):max_index), size = batch_size)
#     } else {
#       if (i + batch_size >= max_index)
#         i <<- min_index + lookback
#       rows <- c(i:min(i+batch_size, max_index))
#       i <<- i + length(rows)
#     }
#     samples <- array(0, dim = c(length(rows),
#                                 lookback / step,
#                                 dim(data)[[-1]]))
#     targets <- array(0, dim = c(length(rows)))
#     for (j in 1:length(rows)) {
#       indices <- seq(rows[[j]] - lookback, rows[[j]],
#                      length.out = dim(samples)[[2]])
#       samples[j,,] <- data[indices,]
#       targets[[j]] <- data[rows[[j]] + delay,predseries]
#     }
#     list(samples, targets)
#   }
# }


# lookback <- 48 #4 years of past data used in each current prediction
# step <- 1 #Do not skip observations (could set to higher value if you had extremely high frquency data, like for stock market data)
# delay <- 1 #Predict 1 month ahead
# batch_size <- 10 #draw 20 samples at a time
# predser <- 6
# 
# train_gen <- generator(
#   data,
#   lookback = lookback,
#   delay = delay,
#   min_index = 1,
#   max_index = 521,
#   shuffle = TRUE,
#   step = step,
#   batch_size = batch_size,
#   predseries = predser  
# )
# 
# val_gen = generator(
#   data,
#   lookback = lookback,
#   delay = delay,
#   min_index = 522,
#   max_index = 640,
#   step = step,
#   batch_size = batch_size,
#   predseries = predser  
# )
# 
# test_gen <- generator(
#   data,
#   lookback = lookback,
#   delay = delay,
#   min_index = 641,
#   max_index = NULL,
#   step = step,
#   batch_size = batch_size,
#   predseries = predser    
# )
# 
# val_steps <- (640 - 521 - lookback) / batch_size
# test_steps <- (nrow(data) - 641 - lookback) / batch_size
# 
# 
# library(keras)
# library(dplyr)
# library(tensorflow)
# install_tensorflow()  
# set.seed(78910)
# densemodel <- keras_model_sequential() %>%
#   layer_flatten(input_shape=c(lookback/step,dim(data)[-1])) %>%
#   layer_dense(units=32,activation="relu") %>%
#   layer_dense(units=1)
# 
# 
# densemodel %>% compile(
#   optimizer = "rmsprop",
#   loss="mse"
# )
# 
# 
# 
# target_column <- "avg_buy_price_LR"
# time_series <- na.omit(data[[target_column]])
# 
# normalize <- function(x) (x - min(x)) / (max(x) - min(x))
# normalized_series <- normalize(time_series)
# 
# 
# create_lagged_data <- function(series, lag) {
#   n <- length(series)
#   x <- matrix(NA, nrow = n - lag, ncol = lag)
#   y <- series[(lag + 1):n]
#   for (i in 1:lag) {
#     x[, i] <- series[(lag + 1 - i):(n - i)]
#   }
#   list(x = x, y = y)
# }
# 
# lag <- 10
# data <- create_lagged_data(normalized_series, lag)
# 
# train_size <- round(0.7 * nrow(data$x))
# x_train <- data$x[1:train_size, ]
# y_train <- data$y[1:train_size]
# x_test <- data$x[(train_size + 1):nrow(data$x), ]
# y_test <- data$y[(train_size + 1):nrow(data$x), ]
# 
# x_train_lstm <- array(x_train, dim = c(nrow(x_train), ncol(x_train), 1))
# x_test_lstm <- array(x_test, dim = c(nrow(x_test), ncol(x_test), 1))
# 
# 
# model <- keras_model_sequential() 
# model %>% 
#   layer_lstm(units = 50, input_shape = c(lag, 1), return_sequences = FALSE) %>% 
#   layer_dense(units = 1)


library(dplyr)
library(lubridate)
library(caret)
library(xgboost)
library(forecast)

train <- data %>% filter(DATE < "2020-01-01")
test <- data %>% filter(DATE >= "2020-01-01")

train_x <- train %>% select(-avg_buy_price_LR, -DATE)
train_y <- train$avg_buy_price_LR

test_x <- test %>% select(-avg_buy_price_LR, -DATE)
test_y <- test$avg_buy_price_LR


train_matrix <- as.matrix(train_x)
test_matrix <- as.matrix(test_x)

xgb_model <- xgboost(
  data = train_matrix,
  label = train_y,
  nrounds = 100,
  objective = "reg:squarederror",
  verbose = 1
)

predictions <- predict(xgb_model, test_matrix)

accuracy(predictions, test$avg_buy_price_LR)

results <- test %>%
  mutate(
    predicted = predictions,
    error = avg_buy_price_LR - predicted
  )

ggplot(results, aes(x = DATE)) +
  geom_line(aes(y = avg_buy_price_LR, color = "Actual")) +
  geom_line(aes(y = predicted, color = "Predicted")) +
  labs(title = "Actual vs Predicted", y = "Response Variable") +
  theme_minimal()



data$avg_buy_price_LR

new_resp <- rep(NA, nrow(data))
window <- 30

for(i in 1:nrow(data)){
  
  id <- which.min(abs(data$DATE -  data$DATE[i] - window))
  new_resp[i] <- data$avg_buy_price_LR[id]
}

data$price_30_days_out <- new_resp

cbind.data.frame(data$avg_buy_price_LR, new_resp)
