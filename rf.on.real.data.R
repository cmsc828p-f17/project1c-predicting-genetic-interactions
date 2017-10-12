#!/opt/local/stow/R-3.3.1/bin/Rscript

setwd("/cbcb/project2-scratch/kycheng/project1c-predicting-genetic-interactions")
library(data.table)
library(ranger)

# gi scores, already unique pairs
dat <- fread("data/collins-sc-emap-gis.tsv")
set.names(dat, c("gene1","gene2","gi.score","gi.type"))
# gene to GO term mapping
go.dat <- fread("data/GO.tsv")
go.dat <- go.dat[Gene %in% dat[, c(gene1,gene2)]]
dat <- dat[gene1 %in% go.dat$Gene & gene2 %in% go.dat$Gene]
go.dat <- split(go.dat$Gene, go.dat$Term)
# genes list
gl <- get.genes.list(dat)
# get features (slow)
dat <- cbind(dat, get.features(go.dat, gl))
save(dat, file="data.RData")

# number of samples
N <- nrow(dat)
# labelling of folds for cross-validation
folds <- sample(rep(1:4, ceiling(N/4)), N)
# CV result
cv.result <- lapply(1:4, function(i) {
  train <- folds!=i
  test <- dat[!train]
  set.seed(1)
  rf <- ranger(Score ~ ., data=dat[train, c(-1,-2,-4)], num.trees=500, mtry=floor((length(dat)-4)/3))
  prd <- predictions(predict(rf, data=test))
  test[, .(gene1, gene2, gi.score.real=gi.score, gi.score.prd=prd)]
})
cv.result <- rbindlist(cv.result, idcol="test.fold")
save(cv.result, file="cv.result.RData")
