####Funções
cutoff <- function (prediction, cana) {
  ponto <- seq(0.2,0.8,by=0.001)
  acuracia <- array(NA, dim=length(ponto))
  for (i in 1:length(ponto)) {
    classif <- as.factor(ifelse(prediction > ponto[i], "1", "0"))
    cm <- confusionMatrix(classif, cana, positive="1")
    acuracia[i] <- cm$overall['Accuracy']
  }
  return(ponto[which(acuracia==max(acuracia))])
}


####Programa principal
library(caret)

setwd("C:/Users/Ana/Documents/Programação R/IC")
data <- read.csv("tabelaNAfix2015_2016 - Area6.csv")


####Organizando dados de referência
data$Classif[data$Classif == "CANA"]  <- 1
data$Classif[data$Classif == "OUTROS"] <- 0
data$Classif[data$Classif == "URBANA"]  <- 0

data$Classif <- factor(data$Classif)
Classif <- data$Classif
indices <- c(1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65)
AnoUm <- data[ ,c(16,18,24,25,26,28,34,35,36,38,44,45,46,48,54,55,
                  56,58,64,65,66,68,74,75,76,78,84,85, 86,88,94,95,
                  96,98,104,105,106,108,114,115,116,118,124,125,
                  126,128,134,135,136,138,144,145,146,148,154,155,
                  156,158,164,165,166,168,174,175,176,178,184,185 )]


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
resultados <- data.frame( JA = 0, FE = 0, MA = 0, AB = 0, MI = 0,
                         JU = 0, JL = 0, AG = 0, SE = 0, OU = 0, NO = 0, DE = 0, 
                         JA2 = 0, FE2 = 0, MA2 = 0, AB2 = 0, MI2 = 0, cana = classTest)               


####Calculo de modelos e predição
for (i in indices){
dataFitT <- data.frame(cana = classT,
                       nd = covT[,i],
                       ev = covT[,(i+1)],
                       s1 = covT[,(i+2)],
                       s2 = covT[,(i+3)])
modelo <- glm(classT~nd+ev+s1+s2, family=binomial(link="logit"), data = dataFitT)
dataFitT$prediction <- predict(modelo, newdata=dataFitT, type="response")
p <- cutoff(dataFitT$prediction, dataFitT$cana)

dataFitTest <- data.frame(cana = classTest,
                          nd = covTest[,i],
                          ev = covTest[,(i+1)],
                          s1 = covTest[,(i+2)],
                          s2 = covTest[,(i+3)])
dataFitTest$prediction  <- predict(modelo, newdata=dataFitTest, type="response")
dataFitTest$pred <- ifelse(dataFitTest$prediction > p, "1", "0")
resultados[ ,match(i, indices)] <- as.factor(ifelse(dataFitTest$prediction > p, "1", "0"))
}

confusionMatrix(resultados[ ,10], dataFitTest$cana, positive="1") #70.51%

####Análise de acurácia
AcuracyCorte <- c(65.78,63.59,67.99,68.77,68.80,69.27,69.67,68.04,70.33,70.51,
                  68.97,67.22,65.13,67.91,66.59,69.39,70.07)
mc <- max(AcuracyCorte)
match(mc, AcuracyCorte)
mn <- max(AcuracyNormal)
match(mn, AcuracyNormal)
AcuracyNormal <- c(64.93,63.30,63.57,64.95,68.84,69.36,68.72,68.65,69.83,70.18,
                   68.53,67.16,65.01,64.58,66.19,68.74,70.07)

plot(1:17,AcuracyCorte, type = "l", col = "red", xlab = "Meses", ylab = "Acurácia",
     main = "Comparação de acurácia",) 
  lines(1:17, AcuracyNormal, col = "blue")
  legend('topleft',
         legend = c('Normal','Ponto de Corte'),
         col = c("blue", "red"), lty = 1, lwd=1, pch = 1, 
         bg='transparent', bty = "n")
?points
    