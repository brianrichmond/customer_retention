## Customer Retention Analysis
## Goals:
##    1. Create model to accurately predict retained customers
##    2. Identify key factors that identify retained customers
## Brian Richmond
## Updated 2018-04-25


# Import libraries
library(lattice)  # graphics tools
library(plyr)  # data manipulation
library(dplyr)  # data frame tools
library(jsonlite)  # importing json files
library(caret)  # functions to streamline process for predictive models
library(pscl)  # logit metrics
library(ROCR)  # ROC plots
library(glmnet)  # Regularized glms
library(randomForest)  # random forest tools

# Import json data for analysis of rider retention, rr
rr.df <- fromJSON("dataset.json")

## Cleaning: Change all characters to factors, create "active" variable
rr.df <- rr.df %>% mutate_if(is.character,as.factor)
rr.df <- rr.df %>% mutate_if(is.logical,as.factor)
rr.df <- rr.df %>% mutate_if(is.integer,as.numeric)
rr.df$last_trip_date <- as.Date(rr.df$last_trip_date)
rr.df$signup_date <- as.Date(rr.df$signup_date)
rr.df$active <- as.factor(ifelse(rr.df$last_trip_date>"2014-06-01","active","inactive"))


## Explore data
# Summary of variables
summary(rr.df)
# Explore missing data
print(paste(sum(complete.cases(rr.df)), "cases are complete out of", nrow(rr.df)))
# Plots of some variables
boxplot(rr.df[,sapply(rr.df,is.numeric)], las=2, par(mar=c(10,4,4,2)),
        main="Boxplot with Outliers")
boxplot(trips_in_first_30_days~city,rr.df)
pairs(rr.df[,sapply(rr.df,is.numeric)])
## Exploring relationship btn missing data & 'active'
Av.na <- rr.df[is.na(rr.df$avg_rating_of_driver),]
summary(Av.na)
Av.nona <-rr.df[!is.na(rr.df$avg_rating_of_driver),]
summary(Av.nona)

## Classic Logistic Regression (glm)
# Prepare data
activeData <- subset(rr.df, select = c(1,2,4,5,7:13))  # select subset of variables for model
set.seed(678)
inTrain<-createDataPartition(y=activeData$active, p=0.7,list=FALSE)  # select 70% of rows
train<-activeData[inTrain,]
test<-activeData[-inTrain,]
contrasts(activeData$active)
activeLogit <-glm(active~.,family = binomial(link="logit"),data=train)
summary(activeLogit)
anova(activeLogit,test="Chisq")
pR2(activeLogit)

# Prediction ability, comparing test and training classification error
train.predict <-predict(activeLogit,newdata=subset(train,select = c(1:10)), type="response")
train.predict.results <-ifelse(train.predict>0.5,1,0)
train.tbl = xtabs(~train.predict.results+train$active)  # create table of prediction results
train.tbl
train.acc <-sum(diag(train.tbl)/sum(train.tbl))
print(paste("Train Accuracy:",train.acc))

# Generate and test predictions based on test data
test.predict <-predict(activeLogit,newdata=subset(test,select = c(1:10)), type="response")
test.results <-ifelse(test.predict>0.5,1,0)
tbl = xtabs(~test.results+test$active)
tbl
p.tbl<-prop.table(xtabs(~test.results+test$active))  # create table of proportions
p.tbl
Acc <-sum(diag(tbl)/sum(tbl))
print(paste("Test Accuracy:",Acc))
print(paste("Test Misclassification Error:",1-Acc))

# ROC curve; require ROCR
prd <-prediction(test.predict,test$active)
roc <-performance(prd, measure = "tpr", x.measure = "fpr")
plot(roc)


## Machine Learning Approach
## Generate models for predicting active users using random forests
set.seed(456)
activeForest <-randomForest(active~.,data=train,ntree=100,importance=TRUE,na.action=na.omit)
activeForest

# Predictions based on training data
Ftrain.predict <-predict(activeForest,newdata=subset(train,select = c(1:10)), type="response")
Ftrain.predict.results <-as.numeric(Ftrain.predict)
Ftrain.tbl = xtabs(~Ftrain.predict.results+train$active)
Ftrain.tbl
Ftrain.acc <-sum(diag(Ftrain.tbl)/sum(Ftrain.tbl))
print(paste("Random Forest Training Set Prediction Accuracy:",Ftrain.acc))

# Predictions based on test data
ForestPreds <-predict(activeForest,newdata = test)
ForestPred.results <-as.numeric(ForestPreds)
ForestPred.tbl = xtabs(~ForestPred.results+test$active)
ForestPred.tbl
Fp.tbl<-prop.table(xtabs(~ForestPred.results+test$active))
Fp.tbl
rf.acc <-sum(diag(ForestPred.tbl)/sum(ForestPred.tbl))
print(paste("Random Forest Test Accuracy:",rf.acc))

# ROC curve of test data
Fprd <-prediction(ForestPred.results,test$active)
Froc <-performance(Fprd, measure = "tpr", x.measure = "fpr")
plot(Froc)

# Examine important variables (type 1=mean decrease in accuracy; 2=...in node impurity)
varImpPlot(activeForest,type=1, main="Variable Importance (Accuracy)",
           sub = "Random Forest Model")
varImpPlot(activeForest,type=2, main="Variable Importance (Node Impurity)",
           sub = "Random Forest Model")
var_importance <-importance(activeForest)
var_importance


## Random forest model including cases with missing data (na.roughfix, using medians)
activeForest.narf <-randomForest(active~.,data=train,ntree=100,importance=TRUE,na.action=na.roughfix)
activeForest.narf

# Predictions based on training data
Ftrain.predict.narf <-predict(activeForest.narf,newdata=subset(train,select = c(1:10)), type="response")
Ftrain.predict.narf.results <-as.numeric(Ftrain.predict.narf)
Ftrain.narf.tbl = xtabs(~Ftrain.predict.narf.results+train$active)
Ftrain.narf.tbl
Ftrain.narf.acc <-sum(diag(Ftrain.narf.tbl)/sum(Ftrain.narf.tbl))
print(paste("Random Forest (all cases) Train Accuracy:",Ftrain.narf.acc))

# Predictions based on test data
ForestPreds.narf <-predict(activeForest.narf,newdata = test)
ForestPred.narf.results <-as.numeric(ForestPreds.narf)
ForestPred.narf.tbl = xtabs(~ForestPred.narf.results+test$active)
ForestPred.narf.tbl
Fp.narf.tbl<-prop.table(xtabs(~ForestPred.narf.results+test$active))
Fp.narf.tbl
Acc.narf <-sum(diag(ForestPred.narf.tbl)/sum(ForestPred.narf.tbl))
print(paste("Random Forest (all cases) Test Accuracy:",Acc.narf))

# Examine important variables (type 1=mean decrease in accuracy; 2=...in node impurity)
varImpPlot(activeForest.narf,type=1, main="Variable Importance (Accuracy)",
           sub = "Random Forest Model (all cases)")
varImpPlot(activeForest.narf,type=2, main="Variable Importance (Node Impurity)",
           sub = "Random Forest Model (all cases)")
var_importance.narf <-importance(activeForest.narf)
var_importance.narf

### END ###
