
#Importing the datasets
Train_sal <- read.csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\Naive bayes\\SalaryData_Train.csv")
str(Train_sal)
View(Train_sal)

# recode diagnosis as a factor
Train_sal$Salary <- factor(Train_sal$Salary, levels = c(" <=50K", " >50K"), labels = c("Low", "High"))

# proportion of ham and spam messages
prop.table(table(Train_sal$Salary))

#Importing the test dataset

Test_sal <- read.csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\Naive bayes\\SalaryData_Test.csv")
str(Test_sal)

# recode diagnosis as a factor
Test_sal$Salary <- factor(Test_sal$Salary, levels = c(" <=50K", " >50K"), labels = c("Low", "High"))

View(Test_sal)

# proportion of ham and spam messages
prop.table(table(Test_sal$Salary))

##  Training a model on the data ----
install.packages("e1071")
library(e1071)

## building naiveBayes classifier.

tsal_classifier <- naiveBayes(Train_sal,Train_sal$Salary)
tsal_classifier
### laplace smoothing, by default the laplace value = 0
## naiveBayes function has laplace parameter, the bigger the laplace smoothing value, 
# the models become same.

tsal_lap <- naiveBayes(Train_sal,Train_sal$Salary,laplace = 3)
tsal_lap

# Evaluating model performance after applying laplace smoothing
Test_sal_pred_lap <- predict(tsal_lap, Test_sal)

##  Evaluating model performance with out laplace
Test_sal_pred <-predict(tsal_classifier,Test_sal)
Test_sal_pred


## crosstable without laplace
install.packages('gmodels')
library(gmodels)

CrossTable(Test_sal_pred, Test_sal$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
## test accuracy
test_acc <- mean(Test_sal_pred == Test_sal$Salary)
test_acc

## crosstable of laplace smoothing model
CrossTable(Test_sal_pred_lap, Test_sal$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

# train accuracy after laplace
train_acc_lap = mean(Train_sal_pred_lap == Train_sal$Salary)
train_acc_lap

# prediction on train data for laplace model
Train_sal_pred_lap<- predict(tsal_lap,Train_sal)
Train_sal_pred_lap

## test accuracy after laplace 
test_acc_lap <- mean(Test_sal_pred_lap == Test_sal$Salary)
test_acc_lap

# On Training Data without laplace 
Train_sal_pred <- predict(tsal_classifier, Train_sal)
Train_sal_pred

# prediction on train data for laplace model
Train_sal_pred_lap<- predict(tsal_lap,Train_sal)
Train_sal_pred_lap










































