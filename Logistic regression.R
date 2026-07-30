#############################################################
### WEIGHTED LOGISTIC REGRESSION CLASSIFICATION
### FARS Fatal Crash Analysis
#############################################################

# Install packages (Run once)
install.packages(c("caret", "pROC", "ggplot2"))

# Load libraries
library(caret)
library(pROC)
library(ggplot2)

#############################################################
# 1. Load Dataset
#############################################################

accident.df <- read.csv("FARS2023.csv")

View(accident.df)

#############################################################
# 2. Create Dependent Variable
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

accident.df[factor.cols] <- lapply(
  accident.df[factor.cols],
  factor
)

#############################################################
# 6. Remove Rare Categories
#############################################################

for(i in factor.cols){
  
  frequency <- table(accident.df[[i]])
  
  valid.levels <- names(
    frequency[frequency >= 2]
  )
  
  accident.df <- accident.df[
    accident.df[[i]] %in% valid.levels,
  ]
  
  accident.df[[i]] <- droplevels(
    accident.df[[i]]
  )
}

#############################################################
# 7. Check Data
#############################################################

dim(accident.df)

table(accident.df$Fatality_Class)

#############################################################
# 8. Train-Test Split
#############################################################

set.seed(123)

train.index <- createDataPartition(
  accident.df$Fatality_Class,
  p = 0.80,
  list = FALSE
)

train.df <- accident.df[train.index, ]

test.df <- accident.df[-train.index, ]

#############################################################
# 9. Match Factor Levels
#############################################################

for(i in factor.cols){
  
  test.df[[i]] <- factor(
    test.df[[i]],
    levels = levels(train.df[[i]])
  )
  
}

test.df <- na.omit(test.df)

#############################################################
# 10. Compute Class Weights
#############################################################

class.counts <- table(train.df$Fatality_Class)

weight.minority <- as.numeric(class.counts["0"] /
                                class.counts["1"])

train.weights <- ifelse(
  train.df$Fatality_Class == "1",
  weight.minority,
  1
)

#############################################################
# 11. Build Weighted Logistic Regression
#############################################################

logistic.model <- glm(
  Fatality_Class ~ .,
  data = train.df,
  family = binomial,
  weights = train.weights
)

#############################################################
# 12. Model Summary
#############################################################

summary(logistic.model)

#############################################################
# 13. Odds Ratios
#############################################################

odds.ratio <- exp(coef(logistic.model))

print(odds.ratio)

#############################################################
# 14. Predict Probabilities
#############################################################

logistic.prob <- predict(
  logistic.model,
  newdata = test.df,
  type = "response"
)

#############################################################
# 15. Classification Threshold
#############################################################

threshold <- 0.5

logistic.pred <- ifelse(
  logistic.prob >= threshold,
  1,
  0
)

logistic.pred <- factor(
  logistic.pred,
  levels = c(0,1)
)

#############################################################
# 16. Confusion Matrix
#############################################################

cm <- confusionMatrix(
  logistic.pred,
  test.df$Fatality_Class,
  positive = "1"
)

print(cm)

#############################################################
# 17. ROC Curve and AUC
#############################################################

roc.model <- roc(
  response = test.df$Fatality_Class,
  predictor = logistic.prob
)

plot(
  roc.model,
  col = "blue",
  lwd = 2,
  main = "ROC Curve - Weighted Logistic Regression"
)

auc.value <- as.numeric(auc(roc.model))

#############################################################
# 18. Performance Metrics
#############################################################

accuracy <- cm$overall["Accuracy"]
kappa <- cm$overall["Kappa"]

sensitivity <- cm$byClass["Sensitivity"]
specificity <- cm$byClass["Specificity"]

cat("\n=======================================\n")
cat("Weighted Logistic Regression Results\n")
cat("=======================================\n")

cat("Accuracy    :", round(accuracy,4), "\n")
cat("AUC         :", round(auc.value,4), "\n")
cat("Sensitivity :", round(sensitivity,4), "\n")
cat("Specificity :", round(specificity,4), "\n")
cat("Kappa       :", round(kappa,4), "\n")

#############################################################
# 19. Save Predictions
#############################################################

test.df$Probability <- logistic.prob
test.df$Prediction <- logistic.pred

View(test.df)

#############################################################
### END OF WEIGHTED LOGISTIC REGRESSION
#############################################################