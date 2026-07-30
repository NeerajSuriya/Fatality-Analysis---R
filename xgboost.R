install.packages(c("xgboost", "caret", "Matrix" ))

library(xgboost)
library(caret)
library(Matrix)

#dataset)
crash.df <- read.csv("FARS2023.csv")

#target variable 
crash.df$Fatality_Class <- ifelse(crash.df$FATALS >= 2, 1, 0)
table(crash.df$Fatality_Class)

#independent variables selected
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

#removing missing ones
crash.df <- na.omit(crash.df)


#categoical to numerical
cat.cols <- c(
  "STATE","COUNTY","CITY","RUR_URB",
  "MONTH","DAY_WEEK",
  "ROUTE","FUNC_SYS","RD_OWNER","NHS",
  "RELJCT1","RELJCT2","TYP_INT",
  "REL_ROAD","WRK_ZONE",
  "LGT_COND","WEATHER","SCH_BUS"
)

for(col in cat.cols){
  crash.df[[col]] <- as.integer(as.factor(crash.df[[col]]))
}

#train and valid
set.seed(123)

train.index <- sample(1:nrow(crash.df),
                      0.6*nrow(crash.df))

train.df <- crash.df[train.index, ]
valid.df <- crash.df[-train.index, ]


#class weight
neg <- sum(train.df$Fatality_Class == 0)
pos <- sum(train.df$Fatality_Class == 1)

scale_pos_weight <- neg / pos
scale_pos_weight


#xgboost matrix
dtrain <- xgb.DMatrix(
  data = as.matrix(train.df[, -1]),
  label = train.df$Fatality_Class
)

dvalid <- xgb.DMatrix(
  data = as.matrix(valid.df[, -1]),
  label = valid.df$Fatality_Class
)


#model params
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 5,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = scale_pos_weight
)

#train model 
xgb.model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, valid = dvalid),
  verbose = 1
)

#predict
prob <- predict(xgb.model, dvalid)

pred.class <- ifelse(prob > 0.5, 1, 0)

#confusion matix
confusionMatrix(
  as.factor(pred.class),
  as.factor(valid.df$Fatality_Class)
)

#variable importance
importance <- xgb.importance(
  feature_names = colnames(train.df[, -1]),
  model = xgb.model
)

importance

#graph
xgb.plot.importance(importance)



