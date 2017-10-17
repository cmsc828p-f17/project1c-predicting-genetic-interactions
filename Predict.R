#This script predicts genetic interations using GO realtions as features.
#Sanju Sinha

#Loading required Libraries
require(ranger)
require(RcppCNPy)
require(statar)
require(doMC)
require(foreach)
require(tictoc)
#Functions Required


#Loading Files needed
GI=read.table('/cbcb/project2-scratch/sanju/project1c-predicting-genetic-interactions-master/Real_data/collins-sc-emap-gis.tsv', sep='\t', header=T)
GO=read.csv('/cbcb/project2-scratch/sanju/project1c-predicting-genetic-interactions-master/Real_data/GO.csv', header=T)
GO_List=split(GO$Gene, GO$Term)

#Number of Genes availble for analysis:: 6046 ; # of Go terms: 5125
feature_vector_fora_pair <- function(GA, GB){
	sapply(GO_List, function(x) sum(!is.na(match(x, c(paste(GA), paste(GB))))))
}

Pairs=GI[,1:2]

#FOllowing will take too long for vector generation. Hence, used the saved one from another script.
##############################################################################################################################################################################################
#A feature vector for each pair.
#tic('Making Feature Vector')
#FV = mclapply(1:nrow(FV), function(x) feature_vector_fora_pair(Pairs[x, 1], Pairs[x, 2]), mc.cores= detectCores())
#FV = do.call(rbind, FV)
#toc()
#############################################################################################################################################################################################
FV=readRDS('/cbcb/project2-scratch/sanju/project1c-predicting-genetic-interactions-master/Real_data/FV.RDS')

#Feature vector for every pair
Input_RF=data.frame(Score_List = GI[1:1000,3], FV = FV[1:1000,])

samp <- sample(nrow(Input_RF), 0.75 * nrow(Input_RF))
train <- Input_RF[samp, ]
test <- Input_RF[-samp, ]

#Number of combination:: Pairs=(99*100/2)= 4950
#RF_Model
tic('making Ranger Forest')
Ranger_forest=ranger(Score_List~., train, num.tree=300, num.threads=64)
toc()
#Prediction using the above mdoel
tic('Ranger Forest prediction')
pred <- predict(Ranger_forest, data = test, num.tree=300, num.threads=64 )
toc()
##Pearson Co-relation
pear.cor= cor.test(test$Score_List, pred$predictions, method='pearson')
Regression_Result=paste('with a', 'p-value < 2.2e-16', 'our prediction has pearson co-relation(without any qq-plot check) of', pear.cor$estimate)
saveRDS(Ranger_forest, '/cbcb/project2-scratch/sanju/project1c-predicting-genetic-interactions-master/Analysis/Ranger_Forest_GO.RDS')
saveRDS(Regression_Result, '/cbcb/project2-scratch/sanju/project1c-predicting-genetic-interactions-master/Analysis/Regression_result.RDS')
##############################################################################################################################################################################################
#________________Let's use Random forest as Binary classifier for interaction method______________#
## ***Hyperparameter to play around with.****
K=2
##
xtile_Input_RF=cbind(Score_List=as.factor(xtile(Input_RF$Score_List, K)), Input_RF[,-1])
samp <- sample(nrow(xtile_Input_RF), 0.75 * nrow(xtile_Input_RF))
xtile_train <- xtile_Input_RF[samp, ]
xtile_test <- xtile_Input_RF[-samp, ]
#Number of combination:: Pairs=(99*100/2)= 4950
#RF_Model
tic('Classification')
xtile_RF_Model=ranger(Score_List~., xtile_train, num.tree=300, num.threads=64 )
toc()
#Prediction using the above mdoel
tic('Class Prediction')
xtile_pred <- predict(xtile_RF_Model, data = xtile_test, num.tree=300, num.threads=64 )
toc()

##Truth box
Classification_Result= table(xtile_pred$predictions, xtile_test$Score_List)

saveRDS(Classification_Result, '/cbcb/project2-scratch/sanju/project1c-predicting-genetic-interactions-master/Analysis/Classification_result.RDS')
