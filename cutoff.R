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