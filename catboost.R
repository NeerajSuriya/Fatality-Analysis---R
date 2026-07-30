############################################################
# CATBOOST CLASSIFICATION - FARS DATASET
############################################################

#Install packages (run once)
install.packages(c("catboost","caret","pROC"))

library(catboost)
library(caret)
library(pROC)

############################################################
# Read Dataset
############################################################

fars.df <- read.csv("FARS_2023.csv")

############################################################
# Target Variable
############################################################

fars.df$Target <- factor(fars.df$Target,
                         levels = c(0,1),
                         labels = c("0","1"))

############################################################
# Train / Validation Split (60:40)
############################################################

set.seed(123)

train.index <- sample(1:nrow(fars.df),
                      0.6*nrow(fars.df))

train.df <- fars.df[train.index, ]
valid.df <- fars.df[-train.index, ]

############################################################
# Class Weights
############################################################

class.counts <- table(train.df$Target)

class.weights <- c(
  "0" = 1,
  "1" = as.numeric(class.counts["0"] /
                     class.counts["1"])
)

print(class.weights)

############################################################
# Convert Target to Numeric
############################################################

train.df$Target <- as.numeric(as.character(train.df$Target))
valid.df$Target <- as.numeric(as.character(valid.df$Target))

############################################################
# Observation Weights
############################################################

train.weights <- ifelse(train.df$Target == 1,
                        class.weights["1"],
                        class.weights["0"])

############################################################
# Create CatBoost Pools
############################################################

train.pool <- catboost.load_pool(
  data = train.df[, !(names(train.df) %in% "Target")],
  label = train.df$Target,
  weight = train.weights
)

valid.pool <- catboost.load_pool(
  data = valid.df[, !(names(valid.df) %in% "Target")],
  label = valid.df$Target
)

############################################################
# Train CatBoost Model
############################################################

params <- list(
  
  loss_function = "Logloss",
  
  eval_metric = "AUC",
  
  iterations = 300,
  
  learning_rate = 0.10,
  
  depth = 6,
  
  random_seed = 123,
  
  verbose = 50
  
)

cat.model <- catboost.train(
  learn_pool = train.pool,
  test_pool = valid.pool,
  params = params
)

############################################################
# Predict Probabilities
############################################################

cat.prob <- catboost.predict(
  cat.model,
  valid.pool,
  prediction_type = "Probability"
)

############################################################
# Convert Probability to Class
############################################################

cat.pred <- ifelse(cat.prob >= 0.5,1,0)

############################################################
# Confusion Matrix
############################################################

cm <- confusionMatrix(
  factor(cat.pred,levels=c(0,1)),
  factor(valid.df$Target,levels=c(0,1))
)

print(cm)

############################################################
# ROC Curve
############################################################

roc.cat <- roc(valid.df$Target,
               cat.prob)

plot(roc.cat,
     col="blue",
     lwd=2,
     main="CatBoost ROC Curve")

auc.cat <- auc(roc.cat)

cat("\nAUC =", auc.cat,"\n")

############################################################
# Feature Importance
############################################################

importance <- catboost.get_feature_importance(
  cat.model,
  pool=train.pool
)

importance.df <- data.frame(
  
  Feature = names(train.df)[names(train.df)!="Target"],
  
  Importance = importance
  
)

importance.df <- importance.df[
  order(-importance.df$Importance),
]

print(head(importance.df,20))

############################################################
# Final Performance Table
############################################################

precision <- cm$byClass["Pos Pred Value"]

recall <- cm$byClass["Sensitivity"]

f1 <- 2*((precision*recall)/(precision+recall))

results <- data.frame(
  
  Model="CatBoost",
  
  Accuracy=cm$overall["Accuracy"],
  
  Precision=precision,
  
  Recall=recall,
  
  F1=f1,
  
  Specificity=cm$byClass["Specificity"],
  
  AUC=as.numeric(auc.cat)
  
)

print(results)