library(ggplot2) #for ggplot function
library(party) #for ctree function
library(e1071) #for naiveBayes function
library(tidyr) #for gather function
library(dplyr) #for group by function
library(neuralnet)
library(nnet)
library(ROCR)
library(NeuralNetTools)

students <- read.csv("E:/MMU/Sem 9/Data Mining/student/student-mat.csv", header=TRUE, sep=";")

#remove unnecessary columns
students$school<-NULL
students$age<-NULL
students$sex<-NULL
students$famsize<-NULL
students$address<-NULL
students$traveltime<-NULL
students$reason<-NULL

#exploratory analysis

dim(students)
nrow(students)
ncol(students)

str(students)
names(students)
attributes(students)
students[1:15,]

summary(students)

#calculate pass or fail variable
Pass <- ifelse(students$G3>8,'PASS','FAIL')
students <- data.frame(students,Pass)

#factorize continuous predictor variables
students$Medu <- factor(students$Medu)

summary(students)

ggplot(students, aes(x = Pass)) + geom_bar(fill="#5EA2CC")

#decision tree to predict pass or fail (G1+G2+G3)
formula <- Pass ~ G1 + G2 + G3
tree <- ctree(formula, data=students)

print(tree)
plot(tree, type="simple")

sprintf('Errors-on-predictions Matrix')
table(predict(tree, newdata=students), students$Pass,dnn=c('Predicted','Actual'))

df.confmatrix <- data.frame(table(predict(tree, newdata=students), students$Pass,dnn=c('Predicted','Actual')))

data_long <- gather(df.confmatrix, Type, Status, Predicted:Actual)

data_long <- data_long %>% group_by(Status,Type) %>% summarise(Frequency=sum(Freq))
ggplot(data_long, aes(x=Status,y=Frequency,fill=Type)) + geom_bar(stat='identity', position='dodge') 

cm<-as.matrix(table(predict(tree, newdata=students), students$Pass,dnn=c('Predicted','Actual')))
n<-sum(cm)
diag<-diag(cm)
diag
accuracy<-sum(diag)/n
accuracy

nc<-nrow(cm)
rowsums<-apply(cm,1,sum)
colsums<-apply(cm,2,sum)
p<-rowsums/n
q<-rowsums/n

precision<-diag/colsums
recall<-diag/rowsums
f1<-2*precision*recall/(precision+recall)

data.frame(precision,recall,f1)

#Naive bayes for predicting pass or fail
classifier<-naiveBayes(students[,1:8], students[,12])
classifier
sprintf('Errors-on-predictions Matrix')

table(predict(classifier, students[,1:8]), students[,12], dnn = c('Predicted','Actual'))

df.confmatrix <- data.frame(table(predict(classifier, students[,1:8]), students[,12], dnn = c('Predicted','Actual')))
df.confmatrix
data_long <- gather(df.confmatrix, Type, Status, Predicted:Actual)
data_long <- data_long %>% group_by(Status,Type) %>% summarise(Frequency=sum(Freq))
ggplot(data_long, aes(x=Status,y=Frequency,fill=Type)) + geom_bar(stat='identity', position='dodge')

# To know the accuracy, precision, recall and f1 based on confusion matrix of Naive Bayes
cm<-as.matrix(table(predict(classifier, students[,1:8]), students[,12], dnn = c('Predicted','Actual')))
n<-sum(cm)
diag<-diag(cm)
diag
accuracy<-sum(diag)/n
accuracy

nc<-nrow(cm)
rowsums<-apply(cm,1,sum)
colsums<-apply(cm,2,sum)
p<-rowsums/n
q<-rowsums/n

precision<-diag/colsums
recall<-diag/rowsums
f1<-2*precision*recall/(precision+recall)

data.frame(precision,recall,f1)    

#ANN
data_train<-students[1:350,]
data_test<-students[351:395,]

net_model <-neuralnet(G3~ G1+G2+goout+Medu+Fedu+absences+failures,
                      data=data_train,hidden=1,linear.output=TRUE)
print(net_model)
plotnet(net_model)

model_results <-compute(net_model,data_test[c("G1","G2","goout","absences",
                                              "failures","Fedu","Medu")])
predicted_G3 <- model_results$net.result
cor(predicted_G3,data_test$G3)[,1]

plot(predicted_G3,data_test$G3, main="1 hidden node layer",
     ylab="Real G3")
abline(a=0,b=1, col="black")
