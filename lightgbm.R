#=========================================================
# LIGHTGBM MODEL - FARS 2023
#=========================================================

# Install packages (run only once)
install.packages("lightgbm")
install.packages("Matrix")
install.packages("caret")
install.packages("pROC")

# Load libraries
library(lightgbm)
library(Matrix)
library(caret)
library(pROC)

#=========================================================
# Load Dataset
#=========================================================

crash.df <- read.csv("FARS2023.csv")

#=========================================================
# Create Target Variable
#=========================================================

crash.df$Fatality_Class <- ifelse(crash.df$FATALS >= 2, 1, 0)

table(crash.df$Fatality_Class)

#=========================================================
# Select Variables
#=========================================================

crash.df <- crash.df[, c(
  "Fatality_Class",
  "STATE",
  "COUNTY",
  "CITY",
  "LATITUDE",
  "LONGITUD",
  "RUR_URB",
  "MONTH",
  "DAY_WEEK",
  "HOUR",
  "MINUTE",
  "ROUTE",
  "FUNC_SYS",
  "RD_OWNER",
  "NHS",
  "RELJCT1",
  "RELJCT2",
  "TYP_INT",
  "REL_ROAD",
  "WRK_ZONE",
  "LGT_COND",
  "WEATHER",
  "SCH_BUS"
)]

#=========================================================
# Remove Missing Values
#=========================================================

crash.df <- na.omit(crash.df)

#=========================================================
# Convert Categorical Variables to Numeric
#=========================================================

cat.cols <- c(
  "STATE",
  "COUNTY",
  "CITY",
  "RUR_URB",
  "MONTH",
  "DAY_WEEK",
  "ROUTE",
  "FUNC_SYS",
  "RD_OWNER",
  "NHS",
  "RELJCT1",
  "RELJCT2",
  "TYP_INT",
  "REL_ROAD",
  "WRK_ZONE",
  "LGT_COND",
  "WEATHER",
  "SCH_BUS"
)

for(col in cat.cols){
  crash.df[[col]] <- as.integer(as.factor(crash.df[[col]]))
}

#=========================================================
# Train-Test Split
#=========================================================

set.seed(123)

train.index <- sample(
  1:nrow(crash.df),
  0.6 * nrow(crash.df)
)

train.df <- crash.df[train.index, ]
valid.df <- crash.df[-train.index, ]

#=========================================================
# Predictor Variables
#=========================================================

predictor_cols <- setdiff(names(train.df), "Fatality_Class")

#=========================================================
# Create Matrices
#=========================================================

x_train <- as.matrix(train.df[, predictor_cols])
y_train <- train.df$Fatality_Class

x_valid <- as.matrix(valid.df[, predictor_cols])
y_valid <- valid.df$Fatality_Class

#=========================================================
# Handle Class Imbalance
#=========================================================

neg <- sum(y_train == 0)
pos <- sum(y_train == 1)

scale_pos_weight <- neg / pos

print(scale_pos_weight)

#=========================================================
# Convert to LightGBM Dataset
#=========================================================

dtrain <- lgb.Dataset(
  data = x_train,
  label = y_train
)

dvalid <- lgb.Dataset(
  data = x_valid,
  label = y_valid
)

#=========================================================
# Model Parameters
#=========================================================

params <- list(
  objective = "binary",
  metric = "auc",
  learning_rate = 0.05,
  num_leaves = 31,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  scale_pos_weight = scale_pos_weight,
  verbosity = -1
)

#=========================================================
# Train Model
#=========================================================

lgb.model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  valids = list(
    train = dtrain,
    valid = dvalid
  ),
  early_stopping_rounds = 10
)

#=========================================================
# Predictions
#=========================================================

pred_prob <- predict(
  lgb.model,
  x_valid
)

#=========================================================
# Convert Probabilities to Classes
#=========================================================

pred_class <- ifelse(pred_prob >= 0.5, 1, 0)

pred_class <- factor(pred_class)
actual <- factor(y_valid)

#=========================================================
# Confusion Matrix
#=========================================================

cm <- confusionMatrix(
  pred_class,
  actual,
  positive = "1"
)

cm

#=========================================================
# Performance Metrics
#=========================================================

accuracy <- cm$overall["Accuracy"]
precision <- cm$byClass["Pos Pred Value"]
recall <- cm$byClass["Sensitivity"]
f1 <- cm$byClass["F1"]

cat("Accuracy :", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall   :", recall, "\n")
cat("F1 Score :", f1, "\n")

#=========================================================
# ROC Curve
#=========================================================

roc_obj <- roc(actual, pred_prob)

plot(
  roc_obj,
  col = "blue",
  lwd = 2,
  main = "ROC Curve - LightGBM"
)

#=========================================================
# AUC
#=========================================================

auc_value <- auc(roc_obj)

cat("AUC:", auc_value, "\n")

#=========================================================
# Feature Importance
#=========================================================

importance <- lgb.importance(
  lgb.model,
  percentage = TRUE
)

print(importance)

lgb.plot.importance(importance)
