####################################################
    ###RANDOM FOREST CLASSIFICATION
####################################################

library(randomForest)
library(caret)
library(ggplot2)

# Load the dataset
accident.df <- read.csv("FARS2023.csv")

View(accident.df)

# Create the target variable
accident.df$Fatality_Class <- ifelse(accident.df$FATALS >= 2, 1, 0)
accident.df$Fatality_Class <- factor(accident.df$Fatality_Class)

# Sample 3000 records
#set.seed(123)
#accident.df <- accident.df[sample(nrow(accident.df), 3000), ]

# Select required variables
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

# Convert categorical variables to factors
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

accident.df[factor.cols] <- lapply(accident.df[factor.cols], factor)

# Remove missing values
accident.df <- na.omit(accident.df)

# Split the dataset into training and testing data
set.seed(123)

train.index <- createDataPartition(
  accident.df$Fatality_Class,
  p = 0.80,
  list = FALSE
)

train.df <- accident.df[train.index, ]
valid.df <- accident.df[-train.index, ]



# Build the Random Forest model
# Number of observations in each class
class_counts <- table(train.df$Fatality_Class)

# Minority class size
min_class <- min(class_counts)

rf.model <- randomForest(
  Fatality_Class ~ .,
  data = train.df,
  ntree = 500,
  mtry = 8,
  importance = TRUE,
  sampsize = c(min_class, min_class)
)


# Display model summary
print(rf.model)

# Predict on test data
rf.pred <- predict(
  rf.model,
  newdata = valid.df
)

# Display first five predictions
data.frame(
  Actual = valid.df$Fatality_Class[1:5],
  Predicted = rf.pred[1:5]
)

# Generate confusion matrix
cm <- confusionMatrix(
  data = rf.pred,
  reference = valid.df$Fatality_Class,
  positive = "1"
)

print(cm)

# Display variable importance values
importance(rf.model)

# Plot variable importance
varImpPlot(
  rf.model,
  main = "Variable Importance Plot"
)

# Plot Random Forest error rate
plot(
  rf.model,
  main = "Random Forest Error Rate"
)

# Create variable importance bar chart
var.imp <- importance(rf.model)

barplot(
  var.imp[, 1],
  las = 2,
  col = "skyblue",
  main = "Variable Importance",
  ylab = "Mean Decrease Accuracy"
)

# Create confusion matrix heatmap
cm.data <- as.data.frame(cm$table)

ggplot(cm.data,
       aes(x = Reference,
           y = Prediction,
           fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq),
            size = 5,
            color = "white") +
  scale_fill_gradient(low = "lightblue",
                      high = "darkblue") +
  labs(title = "Confusion Matrix Heatmap") +
  theme_minimal()

# Plot class distribution
barplot(
  table(accident.df$Fatality_Class),
  col = c("orange", "green"),
  main = "Fatality Class Distribution",
  xlab = "Fatality Class",
  ylab = "Frequency"
)

# Save predictions
valid.df$Prediction <- rf.pred

View(valid.df)












library(caret)
library(pROC)

# Confusion Matrix
cm <- confusionMatrix(
  data = rf.pred,
  reference = valid.df$Fatality_Class,
  positive = "1"
)

# ROC and AUC
rf.prob <- predict(rf.model, valid.df, type = "prob")[,2]
roc.obj <- roc(valid.df$Fatality_Class, rf.prob)
auc.value <- auc(roc.obj)

# Print Metrics
cat("Accuracy    :", round(cm$overall["Accuracy"], 4), "\n")
cat("Sensitivity :", round(cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity :", round(cm$byClass["Specificity"], 4), "\n")
cat("Kappa       :", round(cm$overall["Kappa"], 4), "\n")
cat("AUC         :", round(as.numeric(auc.value), 4), "\n")







table(rf.pred)
print(rf.model)
rf.prob <- predict(rf.model, valid.df, type = "prob")

summary(rf.prob[,2])
cat("AUC :", round(as.numeric(auc.value), 4))





