#### header ####
# This script is NOT a necessary part of the creat-forecast workflow. It checks the data using 
# the package dataMaid and generates a pdf report (opt. for a subset of cases). It also prints
# the dynamics for patients with large creatinine change (>=factor 3)

# 
# Input:
#    + ../results/tomorrows_value_final.csv : final dataset
#
# Output:
#    + pdf output from data maid 
#    + pdf with creatinine curves from icustays with large creatinine change
#
# Author: Andre Beinrucker TMO


####initialize workspace ####
rm(list=ls())
graphics.off()

#packages
#adapt working directory here
setwd("C:/Users/andre.beinrucker/Downloads/MIMIC/programs")
#if you do not have checkpoint installed, you need to run: 
#install.packages(checkpoint)
library(checkpoint)
#all further packages will be installed by checkpoint, set the location:
#checkpoint("2017-08-14", checkpointLocation = "I:/Bereichsaustausch/Biostatistics/Software/StatisticsTools/R/checkpoint_library", R.version="3.1.2")
checkpoint("2017-08-14", checkpointLocation = "C:/Users/andre.beinrucker/Downloads/checkpoint_library", R.version="3.1.2")

library(dataMaid)

#parameters
#choose number of patients to include in dataMaids pdf report, set Inf to include all
nbPatientsToCheck <- 10 

#load labeled data
d <- read.csv(file="../results/tomorrows_value_final.csv", stringsAsFactors = FALSE)
#d <- dFinal


if (nbPatientsToCheck == Inf){ 
  clean(d)
} else {
    clean(d[1:nbPatientsToCheck, ])
  }

pdf("../results/patient_curve_final.pdf")
#get max creatinine for plot
ymax <- 0
#nbPatientsToPlot <- 100
for (pat in unique(d$icustay_id)[1:nbPatientsToPlot])
{
  ymax <- max(ymax, max(d[d$icustay_id==pat, "valuenum"]))
}
#plot evolution for icustays with change of creatinine of at least factor 3
for (icustay in unique(d$icustay_id)[1:nbPatientsToPlot])
{
  firstDate <- min(d[d$icustay_id==icustay, "charttime"])
  
  if (max(d[d$icustay_id==icustay, "valuenum"]) /
      min(d[d$icustay_id==icustay, "valuenum"]) >= 3)
  {
    plot(
      #nb days since first creatinine value 
      difftime(d[d$icustay_id==icustay, "charttime"]
               #[1:min(10, length(d[d$icustay_id==icustay, "charttime"]))]
               , 
               firstDate, unit="days"), 
      #creatinine values
      d[d$icustay_id==icustay, "valuenum"]
      #[1:min(10, length(d[d$icustay_id==icustay, "charttime"]))]
      ,
      xlab="days after first creatinine draw",
      ylab="creatinine value mg/dL",
      ylim=c(0.1,ymax),
      log="y"
    )
    lines(difftime(d[d$icustay_id==icustay, "charttime"]
                   [1:min(10, length(d[d$icustay_id==icustay, "charttime"]))], 
                   firstDate, unit="days"),
          d[d$icustay_id==icustay, "valuenum"]
          [1:min(10, length(d[d$icustay_id==icustay, "charttime"]))],
          xlab="days after first creatinine draw",
          ylab="creatinine value mg/dL")
  }
}
dev.off()

