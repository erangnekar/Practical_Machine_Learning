#Getting and Cleaning Data
## Loading Data 
datTrain <- read.csv("pml-training.csv")
datTest <- read.csv("pml-testing.csv")

##Splitting the data
### Set 60% of the data to be used to build the model, and 40% to test

library(caret)
set.seed(1234)
train <- createDataPartition(y=datTrain$classe, p= .60, list = F)
training <- datTrain[train,]
testing <- datTrain[-train,]

#Cleaning Data - Removing NAs
##Exclude columns that cannot be used for prediction
badCol <- grep("name|timestamp|window|X", colnames(training), value = F)
training <- training[,-badCol]
## Removing columns that are 95% NA
training[training==""] <- NA
rateNA <- apply(training, 2, function(x) sum(is.na(x)))/nrow(training)
training <- training[!(rateNA > .95)]
summary(training)

#PCA
## When data has over 50 variables, we need to run a Principle Component Analysis to narrow it down some more

pca <- prcomp(training[,1:52], scale = TRUE)
summary(pca)
## 25 components are required

pca <- preProcess(training[,1:52], method = 'pca', pcaComp = 25)

trainingPCA <- predict(pca, training[,1:52])

#Random Forest

library(randomForest)
library(e1071)
fitRF <- randomForest(training$classe ~ ., data=trainingPCA, do.trace = F)
print(fitRF)

importance(fitRF)

#Compare with test set

testing2 <- testing[, -badCol]
testing2[testing2 == ""] <- NA
rateNA2 <- apply(testing2, 2, function(x) sum(is.na(x)))/nrow(testing2)
testing2 <- testing2[!(rateNA2 > .95)]
testingPCA <- predict(pca, testing2[,1:52])
confusionMatrix(testing2$classe, predict(fitRF, testingPCA))

#Predict the classe value of the 20 data

datTest2 <- datTest[,-badCol]
datTest2[datTest2 == ""] <- NA
rateNA3 <- apply(datTest2, 2, function(x) sum(is.na(x)))/nrow(datTest2)
datTest2 <- datTest2[!(rateNA3 > .95)]
testPCA <- predict(pca, datTest2[,1:52])
datTest2$classe <- predict(fitRF, testPCA)
print(datTest2$classe)

