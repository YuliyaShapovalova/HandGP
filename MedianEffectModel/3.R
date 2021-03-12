setwd( "/Users/Yuliya/Dropbox/Research/VU/GPDI")
# Changed: run from terminal R CMD SHLIB logistic_growth_BI_GPDI_full.c
#-----------------------------------------------------------------------------------
#
# Exploration of the dataset in Molecular Systems Biology (2011) 7: 544
# Systematic exploration of synergistic drug pairs
# by Murat Cokol et al.
#
# Application of GPDI model on the dataset
#
# Author: Sebastian Wicha, Uppsala Universitet
#
# date: 2016-04-21
#
# simplified example script - 2019-01-03
#-----------------------------------------------------------------------------------

require(reshape2)
require(ggplot2)
require(deSolve)

datafiles = "Cis-Cis.txt"

#- read datasets into R -------------------------------------------------------------
# lengths datafiles == 1
for (i in 1:length(datafiles)) {
  assign(paste("Int.",gsub("-",".",datafiles[i]),sep = ""), read.table(datafiles[i]))
  print(paste("Reading ",datafiles[i]))
}




#- concentration vectors (relative to MICs) -----------------------------------------
c.drug.A <- seq(0,1,length.out = 8)
c.drug.B <- seq(0,1,length.out = 8)

#- create concentration columns
dummy = NULL
df    = NULL
for (i in 1:length(unique(c.drug.A))) {
  dummy = cbind(c.drug.A[i],c.drug.B)
  df = rbind(df,dummy)
}

#- DRUG A correspons to first mentioned drug, DRUG B to second ----------------------
colnames(df) = c("DRUG_A","DRUG_B")
df = as.data.frame(df)
Conc_cols <- df[rep(seq_len(nrow(df)), each = 96),]

#- extract data for longitudinal modelling in long format ---------------------------
for (i in 1:length(datafiles)) {
  eval(paste("Int.",gsub("-",".",datafiles[i]),sep = "")) #the name Int.Bro.Sta.txt
  eval(parse(text = paste(
    "Int.",gsub("-",".",datafiles[i]),"[,'TIME'] <- seq(0,23.75,.25)",sep =
      ""
  )))
  eval(parse(
    text = paste(
      "Long.",gsub("-",".",datafiles[i])," <- cbind(melt(Int.",gsub("-",".",datafiles[i]),",id.vars = c('TIME')),Conc_cols)",sep =
        ""
    )
  ))
  print(paste("Creating ",datafiles[i]))
}


#- create informative label with drug and concentration as fold-MIC -----------------
for (i in 1:length(datafiles)) {
  current_df <- paste("Long.",gsub("-",".",datafiles[i]),sep = "")
  lab_df <-
    paste("A",round(Long.Cis.Cis.txt$DRUG_A,2), "B",round(Long.Cis.Cis.txt$DRUG_B,2))
  Long.Cis.Cis.txt$LABEL = paste("A",round(Long.Cis.Cis.txt$DRUG_A,2), "B",round(Long.Cis.Cis.txt$DRUG_B,2))
  eval(parse(text = paste(current_df,"[,'LABEL'] <- lab_df",sep = "")))
  
}

write.csv(Long.Cis.Cis.txt, 'Long.Cis.Cis.csv')