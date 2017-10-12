#!/opt/local/stow/R-3.3.1/bin/Rscript

setwd("/cbcb/project2-scratch/kycheng/project1c-predicting-genetic-interactions")
library(data.table)
library(ranger)

load("data.RData") # dat

# number of samples
N <- nrow(dat)
# labelling of folds for cross-validation
folds <- sample(rep(1:4, ceiling(N/4)), N)
# CV result
cv.result <- lapply(1:4, function(i) {
  train <- folds!=i
  test <- dat[!train]
  set.seed(1)
  rf <- ranger(gi.score ~ ., data=dat[train, c(-1,-2,-4)], num.trees=500, mtry=floor((length(dat)-4)/3), num.threads=24L)
  prd <- predictions(predict(rf, data=test))
  test[, .(gene1, gene2, gi.score.real=gi.score, gi.score.prd=prd)]
})
cv.result <- rbindlist(cv.result, idcol="test.fold")
save(cv.result, file="cv.result.RData")
