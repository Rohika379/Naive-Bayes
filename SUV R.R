# Import the dataset
library(readr)
car <- read.csv(file.choose())
View(car)

str(car)

# recode diagnosis as a factor
car$Purchased <- factor(car$Purchased, levels =c("0","1"),labels =c("No","Yes"))

# examine the type variable more carefully
str(car$Purchased)
table(car$Purchased)

#Spliting the data to train and test
Car_train<-car[1:280,]
Car_test <-car[281:400,]

##  Training a model on the data ----
install.packages("e1071")
library(e1071)

## building naiveBayes classifier.
car_classifier <- naiveBayes(Car_train, Car_train$Purchased)
car_classifier

?naiveBayes
### laplace smoothing, by default the laplace value = 0
## naiveBayes function has laplace parameter, the bigger the laplace smoothing value, 
# the models become same.
car_lap <- naiveBayes(Car_train, Car_train$Purchased,laplace = 3)
car_lap

##  Evaluating model performance with out laplace
car_test_pred <- predict(car_classifier, Car_test)

# Evaluating model performance after applying laplace smoothing
car_test_pred_lap <- predict(car_lap, Car_test)

## crosstable without laplace
library(gmodels)

CrossTable(car_test_pred, Car_test$Purchased,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy
test_acc <- mean(car_test_pred == Car_test$Purchased)
test_acc

## crosstable of laplace smoothing model
CrossTable(car_test_pred_lap, Car_test$Purchased,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy after laplace 
test_acc_lap <- mean(car_test_pred_lap == Car_test$Purchased)
test_acc_lap

# On Training Data without laplace 
car_train_pred <- predict(car_classifier, Car_train)
car_train_pred

# train accuracy
train_acc = mean(car_train_pred == Car_train$Purchased)
train_acc

# prediction on train data for laplace model
car_train_pred_lap <- predict(car_lap,Car_train)
car_train_pred_lap

# train accuracy after laplace
train_acc_lap = mean(car_train_pred_lap == Car_train$Purchased)
train_acc_lap
