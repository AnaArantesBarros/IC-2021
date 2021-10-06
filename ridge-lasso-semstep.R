####Função para encontrar valor de corte ótimo para cada mês
cutoff <- function (prediction, cana) {
  ponto <- seq(0.2,0.8,by=0.01)
  acuracia <- array(NA, dim=length(ponto))
  for (i in 1:length(ponto)) {
    classif <- as.factor(ifelse(prediction > ponto[i], "1", "0"))
    cm <- confusionMatrix(classif, cana, positive="1")
    acuracia[i] <- cm$overall['Accuracy']
  }
  return(ponto[which(acuracia==max(acuracia))])
}


####Programa principal, importando data
library(caret)
library(glmnet)
setwd("C:/Users/Ana/Documents/Programação R/IC")
data <- read.csv("tabelaNAfix2015_2016 - Area6.csv")


####Organizando dados de referência
##A predição que nos interessa será classificada como 1, as demais como 0.
data$Classif[data$Classif == "CANA"]  <- 1
data$Classif[data$Classif == "OUTROS"] <- 0
data$Classif[data$Classif == "URBANA"]  <- 0

data$Classif <- factor(data$Classif)
Classif <- data$Classif
##Selecionando as colunas que serão utilizadas
#Neste caso os indices em data iniciam na 18 e se repetem a cada 10 até a coluna 185
colunas <- c(
  seq(from = 16, to = 185, by = 10),
  seq(from = 17, to = 185, by = 10),
  seq(from = 18, to = 185, by = 10),
  seq(from = 19, to = 185, by = 10),
  seq(from = 24, to = 185, by = 10),
  seq(from = 25, to = 185, by = 10),
  186)
AnoUm <- data[ ,c(colunas)]

####Definindo conjunto de treino e de validação
nT <- round(0.8*nrow(AnoUm), 0)
nTest <- round(0.2*nrow(AnoUm), 0)
set.seed(7) #definir semente para a pesquisa ser reproduzível
idxT <- sort(sample(1:nrow(AnoUm), nT)) #sorteia e ordena os índices do treino
idxTest <- setdiff(1:nrow(AnoUm),idxT) #pega os demais índices para teste

covT      <- AnoUm[idxT,]
covTest   <- AnoUm[idxTest,]
classT    <- Classif[idxT] 
classTest <- Classif[idxTest]
####Armazenamento das predições
###Data frame para armazenar as predições por mês e uma coluna com a classificação real
resultados <- data.frame( JA = 0, FE = 0, MA = 0, AB = 0, MI = 0,
                          JU = 0, JL = 0, AG = 0, SE = 0, OU = 0, NO = 0, DE = 0, 
                          JA2 = 0, FE2 = 0, MA2 = 0, AB2 = 0, MI2 = 0, cana = classTest)


####Calculo de modelos e predição
###Data frame com os indices nd, nw, ev, ls, s1 e s1 dos meses 1 até 17.
dataFitT <- data.frame(cana = classT,
                       nd1 = covT[ ,1],nd2= covT[ ,2] ,nd3= covT[ ,3],nd4= covT[ ,4],
                       nd5= covT[ ,5],nd6= covT[ ,6],nd7= covT[ ,7],nd8= covT[ ,8],
                       nd9= covT[ ,9],nd10= covT[ ,10],nd11= covT[ ,11],nd12= covT[ ,12],
                       nd13= covT[ ,13],nd14= covT[ ,14],nd15= covT[ ,15],nd16= covT[ ,16],
                       nd17= covT[ ,17],
                       nw1= covT[,18],nw2= covT[,19],nw3= covT[,20],nw4= covT[,21],
                       nw5= covT[,22],nw6= covT[,23],nw7= covT[,24],nw8= covT[,25],nw9= covT[,26],
                       nw10= covT[,27],nw11= covT[,28],nw12= covT[,29],nw13= covT[,30],nw14= covT[,31],
                       nw15= covT[,32],nw16= covT[,33],nw17= covT[,34],
                       ev1=covT[ ,35],ev2=covT[ ,36],ev3=covT[ ,37],ev4=covT[ ,38],ev5=covT[ ,39],
                       ev6=covT[ ,40],ev7=covT[ ,41],ev8=covT[ ,42],ev9=covT[ ,43],ev10=covT[ ,44],
                       ev11=covT[ ,45],ev12=covT[ ,46],ev13=covT[ ,47],ev14=covT[ ,48],ev15=covT[ ,49],
                       ev16=covT[ ,50],ev17=covT[ ,51],
                       ls1= covT[ ,52],ls2= covT[ ,53],ls3= covT[ ,54],ls4= covT[ ,55],ls5= covT[ ,56],
                       ls6= covT[ ,57],ls7= covT[ ,58],ls8= covT[ ,59],ls9= covT[ ,60],ls10= covT[ ,61],
                       ls11= covT[ ,62],ls12= covT[ ,63],ls13= covT[ ,64],ls14= covT[ ,65],ls15= covT[ ,66],
                       ls16= covT[ ,67],ls17= covT[ ,68],
                       s11=covT[ ,69],s12=covT[ ,70],s13=covT[ ,71],s14=covT[ ,72],s15=covT[ ,73],
                       s16=covT[ ,74],s17=covT[ ,75],s18=covT[ ,76],s19=covT[ ,77],s110=covT[ ,78],
                       s111=covT[ ,79],s112=covT[ ,80],s113=covT[ ,81],s114=covT[ ,82],s115=covT[ ,83],
                       s116=covT[ ,84],s117=covT[ ,85],
                       s21 = covT[ ,86],s22 = covT[ ,87],s23 = covT[ ,88],s24 = covT[ ,89],s25 = covT[ ,90],
                       s26 = covT[ ,91],s27 = covT[ ,92],s28 = covT[ ,93],s29 = covT[ ,94],s210 = covT[ ,95],
                       s211 = covT[ ,96],s212 = covT[ ,97],s213 = covT[ ,98],s214 = covT[ ,99],
                       s215 = covT[ ,100],s216 = covT[ ,101],s217 = covT[ ,102])

##Modelo com os indices escolhidos por stepwise
x <- model.matrix(cana~.,dataFitT)[,-1]
y <- covT$Classif

#Modelo Regressão Logistica sem penalização
mod1 <- glmnet(x, y, alpha = 0, family = "binomial", lambda = 0)
summary(mod1)
#Modelo Regressão Logistica com penalização ridge
mod2 <- cv.glmnet(x, y, family = "binomial", alpha = 0)

#Modelo Regressão Logistica com penalização lasso
mod3 <- cv.glmnet(x, y, family = "binomial", alpha = 1)



#Predição1 Treino (sem penalização)
dataFitT$prediction1 <- predict(mod1, s=0, newx=x, type="response")
p1 <- cutoff(dataFitT$prediction1, dataFitT$cana)
resultados01 <- as.factor(ifelse(dataFitT$prediction1 > p1, "1", "0"))
confusionMatrix(resultados01, dataFitT$cana, positive="1")

#Predição2 Treino (penalização ridge)
dataFitT$prediction2 <- predict(mod2, s=mod2$lambda.1se, newx=x, type="response")
p2 <- cutoff(dataFitT$prediction2, dataFitT$cana)
resultados02 <- as.factor(ifelse(dataFitT$prediction2 > p2, "1", "0"))
confusionMatrix(resultados02, dataFitT$cana, positive="1")

#Predição3 Treino (penalização lasso)
dataFitT$prediction3 <- predict(mod3, s=mod3$lambda.1se, newx=x, type="response")
p3 <- cutoff(dataFitT$prediction3, dataFitT$cana)
resultados03 <- as.factor(ifelse(dataFitT$prediction3 > p3, "1", "0"))
confusionMatrix(resultados03, dataFitT$cana, positive="1")

##Precisamos utilizar um data frame só com os dados teste
dataFitTest <- data.frame(cana = classTest,
                          nd1 = covTest[ ,1],nd2= covTest[ ,2] ,nd3= covTest[ ,3],nd4= covTest[ ,4],
                          nd5= covTest[ ,5],nd6= covTest[ ,6],nd7= covTest[ ,7],nd8= covTest[ ,8],
                          nd9= covTest[ ,9],nd10= covTest[ ,10],nd11= covTest[ ,11],nd12= covTest[ ,12],
                          nd13= covTest[ ,13],nd14= covTest[ ,14],nd15= covTest[ ,15],nd16= covTest[ ,16],
                          nd17= covTest[ ,17],
                          nw1= covTest[ ,18],nw2= covTest[,19],nw3= covTest[,20],nw4= covTest[,21],
                          nw5= covTest[,22],nw6= covTest[,23],nw7= covTest[,24],nw8= covTest[,25],nw9= covTest[,26],
                          nw10= covTest[,27],nw11= covTest[,28],nw12= covTest[,29],nw13= covTest[,30],nw14= covTest[,31],
                          nw15= covTest[,32],nw16= covTest[,33],nw17= covTest[,34],
                          ev1=covTest[ ,35],ev2=covTest[ ,36],ev3=covTest[ ,37],ev4=covTest[ ,38],ev5=covTest[ ,39],
                          ev6=covTest[ ,40],ev7=covTest[ ,41],ev8=covTest[ ,42],ev9=covTest[ ,43],ev10=covTest[ ,44],
                          ev11=covTest[ ,45],ev12=covTest[ ,46],ev13=covTest[ ,47],ev14=covTest[ ,48],ev15=covTest[ ,49],
                          ev16=covTest[ ,50],ev17=covTest[ ,51],
                          ls1= covTest[ ,52],ls2= covTest[ ,53],ls3= covTest[ ,54],ls4= covTest[ ,55],ls5= covTest[ ,56],
                          ls6= covTest[ ,57],ls7= covTest[ ,58],ls8= covTest[ ,59],ls9= covTest[ ,60],ls10= covTest[ ,61],
                          ls11= covTest[ ,62],ls12= covTest[ ,63],ls13= covTest[ ,64],ls14= covTest[ ,65],ls15= covTest[ ,66],
                          ls16= covTest[ ,67],ls17= covTest[ ,68],
                          s11=covTest[ ,69],s12=covTest[ ,70],s13=covTest[ ,71],s14=covTest[ ,72],s15=covTest[ ,73],
                          s16=covTest[ ,74],s17=covTest[ ,75],s18=covTest[ ,76],s19=covTest[ ,77],s110=covTest[ ,78],
                          s111=covTest[ ,79],s112=covTest[ ,80],s113=covTest[ ,81],s114=covTest[ ,82],s115=covTest[ ,83],
                          s116=covTest[ ,84],s117=covTest[ ,85],
                          s21 = covTest[ ,86],s22 = covTest[ ,87],s23 = covTest[ ,88],s24 = covTest[ ,89],s25 = covTest[ ,90],
                          s26 = covTest[ ,91],s27 = covTest[ ,92],s28 = covTest[ ,93],s29 = covTest[ ,94],s210 = covTest[ ,95],
                          s211 = covTest[ ,96],s212 = covTest[ ,97],s213 = covTest[ ,98],s214 = covTest[ ,99],
                          s215 = covTest[ ,100],s216 = covTest[ ,101],s217 = covTest[ ,102])


##Fazendo a predição com o modelo montado anteriormente
x.test <- model.matrix(cana~.,dataFitTest)[,-1]

#Predição1 Teste (sem penalização)
dataFitTest$prediction1 <- predict(mod1, s=0, newx=x.test, type="response")
p1 <- cutoff(dataFitTest$prediction1, dataFitTest$cana)
resultados1 <- as.factor(ifelse(dataFitTest$prediction1 > p1, "1", "0"))
confusionMatrix(resultados1, dataFitTest$cana, positive="1")

#Predição2 Teste (penalização ridge)
dataFitTest$prediction2 <- predict(mod2, s=mod2$lambda.1se, newx=x.test, type="response")
p2 <- cutoff(dataFitTest$prediction2, dataFitTest$cana)
resultados2 <- as.factor(ifelse(dataFitTest$prediction2 > p2, "1", "0"))
confusionMatrix(resultados2, dataFitTest$cana, positive="1")

#Predição3 Teste (penalização lasso)
dataFitTest$prediction3 <- predict(mod3, s=mod3$lambda.1se, newx=x.test, type="response")
p3 <- cutoff(dataFitTest$prediction3, dataFitTest$cana)
resultados3 <- as.factor(ifelse(dataFitTest$prediction3 > p3, "1", "0"))
confusionMatrix(resultados3, dataFitTest$cana, positive="1")

#Criação de banco de dados
dadoscsv <- data.frame(RegLog = as.numeric(resultados01)-1,
                       Ridge = as.numeric(resultados02)-1,
                       Lasso = as.numeric(resultados03)-1)

dadoscsv1 <- data.frame(RegLog = as.numeric(resultados1)-1,
                        Ridge = as.numeric(resultados2)-1,
                        Lasso = as.numeric(resultados3)-1)

mods <- rbind(dadoscsv, dadoscsv1)
mods$OBJECTID <- c(data$OBJECTID[idxT], data$OBJECTID[idxTest])
mods$OBJECTID_1 <- c(data$OBJECTID_1[idxT], data$OBJECTID_1[idxTest])
mods$Classif <- c(data$Classif[idxT], data$Classif[idxTest])
mods$Classif <- as.numeric(mods$Classif)-1


write.table(mods, file="dados-pred-reglog.csv", 
            sep=";", dec=",", row.names=FALSE,
            col.names=c("RegLog","Ridge","Lasso","OBJECTID","OBJECTID_1", "Classif"))
#load necessary packages
library(ggplot2)
library(pROC)

#define object to plot and calculate AUC mod 1
rocobj1 <- roc(as.numeric(dataFitTest$cana), as.numeric(resultados1))
auc1 <- round(auc(as.numeric(dataFitTest$cana), as.numeric(resultados1)),4)
#oq seria o "4" na formula de auc

#define object to plot and calculate AUC mod 2
rocobj2 <- roc(as.numeric(dataFitTest$cana), as.numeric(resultados2))
auc2 <- round(auc(as.numeric(dataFitTest$cana), as.numeric(resultados2)),4)

#define object to plot and calculate AUC mod 1
rocobj3 <- roc(as.numeric(dataFitTest$cana), as.numeric(resultados3))
auc3 <- round(auc(as.numeric(dataFitTest$cana), as.numeric(resultados3)),4)
#create ROC plot

roclist <- list("Regressão Logística" = rocobj1,
                "Regressão com penalização Ridge" = rocobj2,
                "Regressão com penalização Lasso" = rocobj3)

#create ROC plot
ggroc(roclist, aes = "linetype", legacy.axes = TRUE) +
  geom_abline() +
  theme_classic() +
  ggtitle("Curva ROC com todas as variaveis ") +
  labs(x = "1 - Especificidade",
       y = "Sensibilidade",
       linetype = "Modelo")



