#!/opt/local/stow/R-3.3.1/bin/Rscript

setwd("/cbcb/project2-scratch/kycheng/project1c-predicting-genetic-interactions")
library(data.table)

get.genes.list <- function(response.dat) {
  transpose(response.dat[, 1:2])
}

get.features <- function(go, genes.list) {
  as.data.table(lapply(go, function(go.term) {
    as.numeric(sapply(genes.list, function(genes) -sum(genes %in% go.term)))
  }))
}

# gi scores, already unique pairs
dat <- fread("data/collins-sc-emap-gis.tsv")
setnames(dat, c("gene1","gene2","gi.score","gi.type"))
# gene to GO term mapping
go.dat <- fread("data/GO.tsv")
go.dat <- go.dat[Gene %in% dat[, c(gene1,gene2)]]
dat <- dat[gene1 %in% go.dat$Gene & gene2 %in% go.dat$Gene]
go.dat <- split(go.dat$Gene, go.dat$Term)
# genes list
gl <- get.genes.list(dat)
# get features (slow)
dat <- cbind(dat, get.features(go.dat, gl))
# this turns out to be important!!! who knows!!!!
library(stringr)
setnames(dat, str_replace(names(dat), ":", ""))
save(dat, file="data.RData")

#dat <- fread("data/collins-sc-emap-gis.tsv")
#setnames(dat, c("gene1","gene2","gi.score","gi.type"))
## gene to GO term mapping
#go.dat <- fread("data/GO.tsv")
#go.dat <- go.dat[Gene %in% dat[, c(gene1,gene2)]]
#dat <- dat[gene1 %in% go.dat$Gene & gene2 %in% go.dat$Gene]
## cast go.dat into gene by GO term matrix
#go.dat[, flag:=-1]
#go.dat <- dcast(go.dat, Gene~Term)
#go.dat[is.na(go.dat)] <- 0
## remove GO terms that don't map to any gene
#go.dat[, c(names(go.dat)[c(FALSE,sapply(.SD, sum)==0)]):=NULL, .SDcols=-1] # no such GO terms
#dat[, id:=1:.N]
#f1 <- merge(dat[,.(id,gene1)], go.dat, by.x="gene1", by.y="Gene")
#f2 <- merge(dat[,.(id,gene2)], go.dat, by.x="gene2", by.y="Gene")