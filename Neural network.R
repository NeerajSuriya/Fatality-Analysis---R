#############################################################
### NEURAL NETWORK CLASSIFICATION
### FARS Fatal Crash Analysis (Balanced Training Data)
#############################################################

# Install packages (Run once)
install.packages(c("caret","nnet","pROC"))

# Load libraries
library(caret)
library(nnet)
library(pROC)

#############################################################
# 1. Load Dataset
#############################################################

accident.df <- read.csv("FARS2023.csv")

View(accident.df)

#############################################################
# 2. Create Target Variable
#############################################################

accident.df$Fatality_Class <- ifelse(
  accident.df$FATALS >= 2,
  1,
  0
)

accident.df$Fatality_Class <- factor(
  accident.df$Fatality_Class,
  levels = c(0,1)
)

#############################################################
# 3. Select Variables
#############################################################

accident.df <- accident.df[, c(
  
  "STATE",
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
  "SCH_BUS",
  
  "Fatality_Class"
  
)]

#############################################################
# 4. Remove Missing Values
#############################################################

accident.df <- na.omit(accident.df)

#############################################################
# 5. Convert Categorical Variables
#############################################################

factor.cols <- c(
  
  "STATE",
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

accident.df[factor.cols] <-
  lapply(
    accident.df[factor.cols],
    factor
  )

#############################################################
# 6. Train-Test Split
#############################################################

set.seed(123)

train.index <- createDataPartition(
  accident.df$Fatality_Class,
  p = 0.80,
  list = FALSE
)

train.df <- accident.df[train.index, ]
test.df  <- accident.df[-train.index, ]

#############################################################
# 7. Balance Training Data
#############################################################

majority <- train.df[
  train.df$Fatality_Class == "0",
]

minority <- train.df[
  train.df$Fatality_Class == "1",
]

set.seed(123)

minority.up <- minority[
  sample(
    1:nrow(minority),
    nrow(majority),
    replace = TRUE
  ),
]

train.df <- rbind(
  majority,
  minority.up
)

train.df <- train.df[
  sample(1:nrow(train.df)),
]

table(train.df$Fatality_Class)

#############################################################
# 8. Separate Target
#############################################################

train.y <- train.df$Fatality_Class
test.y  <- test.df$Fatality_Class

train.y.numeric <- as.numeric(
  as.character(train.y)
)

#############################################################
# 9. Separate Predictors
#############################################################

train.x.data <- train.df[
  !names(train.df) %in% "Fatality_Class"
]

test.x.data <- test.df[
  !names(test.df) %in% "Fatality_Class"
]

#############################################################
# 10. Dummy Encoding
#############################################################

dummy.model <- dummyVars(
  ~ .,
  data = train.x.data
)

train.x <- predict(
  dummy.model,
  newdata = train.x.data
)

test.x <- predict(
  dummy.model,
  newdata = test.x.data
)

train.x <- as.data.frame(train.x)
test.x  <- as.data.frame(test.x)
#############################################################
# 11. Remove Near Zero Variance Variables
#############################################################

zero.var <- nearZeroVar(train.x)

if(length(zero.var) > 0){
  
  train.x <- train.x[, -zero.var]
  
  test.x <- test.x[, -zero.var]
  
}

#############################################################
# 12. Scale Data
#############################################################

scaler <- preProcess(
  train.x,
  method = c("center","scale")
)

train.x <- predict(
  scaler,
  train.x
)

test.x <- predict(
  scaler,
  test.x
)

#############################################################
# 13. Build Neural Network
#############################################################

set.seed(123)

nn.model <- nnet(
  
  x = train.x,
  
  y = train.y.numeric,
  
  size = 20,
  
  maxit = 1000,
  
  decay = 0.001,
  
  MaxNWts = 10000,
  
  linout = FALSE,
  
  trace = TRUE
  
)

#############################################################
# 14. Predict Probabilities
#############################################################

nn.prob <- predict(
  
  nn.model,
  
  test.x,
  
  type = "raw"
  
)

#############################################################
# 15. Classification Threshold
#############################################################

threshold <- 0.5

nn.pred <- ifelse(
  
  nn.prob >= threshold,
  
  1,
  
  0
  
)

nn.pred <- factor(
  
  nn.pred,
  
  levels = c(0,1)
  
)

#############################################################
# 16. Confusion Matrix
#############################################################

nn.cm <- confusionMatrix(
  
  data = nn.pred,
  
  reference = test.y,
  
  positive = "1"
  
)

print(nn.cm)

#############################################################
# 17. ROC Curve
#############################################################

roc.nn <- roc(
  
  response = test.y,
  
  predictor = as.numeric(nn.prob)
  
)

plot(
  
  roc.nn,
  
  col = "red",
  
  lwd = 2,
  
  main = "ROC Curve - Neural Network"
  
)

auc.value <- as.numeric(
  
  auc(roc.nn)
  
)

#############################################################
# 18. Performance Metrics
#############################################################

accuracy <- nn.cm$overall["Accuracy"]

kappa <- nn.cm$overall["Kappa"]

sensitivity <- nn.cm$byClass["Sensitivity"]

specificity <- nn.cm$byClass["Specificity"]

cat("\n=====================================\n")
cat("Neural Network Results\n")
cat("=====================================\n")
cat("Accuracy    :", round(accuracy,4), "\n")
cat("AUC         :", round(auc.value,4), "\n")
cat("Sensitivity :", round(sensitivity,4), "\n")
cat("Specificity :", round(specificity,4), "\n")
cat("Kappa       :", round(kappa,4), "\n")

#############################################################
# 19. Save Predictions
#############################################################

test.df$Probability <- nn.prob

test.df$Prediction <- nn.pred

View(test.df)

#############################################################
### END OF NEURAL NETWORK MODEL
#############################################################