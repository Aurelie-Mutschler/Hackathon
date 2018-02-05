#### header ####
# This script loads the data prepared by Aurelie and labels them. It also computes the values creatinine_tomorrow, creatinine_yesterday and creatinine_before_yesterday
# 
# Input:
#    + ../data/creatinine_measurements.csv : extract from database
#
# Output:
#    + ../results/tomorrows_value.csv : input + labels
#
#
# Author: Andre Beinrucker TMO

#### initialization ####

rm(list=ls())
graphics.off()

setwd("C:/Users/andre.beinrucker/Downloads/MIMIC/programs")

#### input ####
dc <- read.csv(file="../data/creatinine_measurements.csv", stringsAsFactors = FALSE)

#remove column X
dc <- dc[, -c(1)]
head(dc)

#compute creatinine hours
dc$creatinine_hours <- as.numeric(substr(dc$creatinine_time, 12, 13))
plot(table(dc$creatinine_hours))

#only use creatinine between 0 and 5
dcf <- dc[dc$creatinine_hours>=0 & dc$creatinine_hours<=5, ]

#save old column as creatinine_time_char (as string) 
dcf$creatinine_time_char <- dcf$creatinine_time
#convert creatinine_time to time format (actually we do not need this)
dcf$creatinine_time <- strptime(dcf$creatinine_time, format='%Y-%m-%d %H:%M:%S')

#extract date
dcf$creatinine_date <- as.Date(substr(dcf$creatinine_time_char,1,10))

#loop over all rows of the creatinine table, for each row compute creatinine_tomorrow as the max
#over all rows which have the same icustay_id and tomorrows date
#this loop is very inefficient, takes ~2h
#todo: improve this by using apply() or by extracting for each icustay the relevant rows of the table,
#      make computation for all creatinine values of this stay in small table
for (i in 1:nrow(dcf))
{
  #to track progress:
  if (i == 100) print(i)
  if (i %% 10000 == 0) print(i)
  #compute tomorrows, yesterdays and day before yesterdays value
dcf[i, "creatinine_tomorrow"] <- 
  max(dcf[(dcf$icustay_id == dcf[i, "icustay_id"]) & 
          (dcf$creatinine_date == dcf[i, "creatinine_date"] + 1), 
        "creatinine"])
dcf[i, "creatinine_yesterday"] <- 
  max(dcf[(dcf$icustay_id == dcf[i, "icustay_id"]) & 
            (dcf$creatinine_date == dcf[i, "creatinine_date"] - 1), 
          "creatinine"])
dcf[i, "creatinine_before_yesterday"] <- 
  max(dcf[(dcf$icustay_id == dcf[i, "icustay_id"]) & 
            (dcf$creatinine_date == dcf[i, "creatinine_date"] - 2), 
          "creatinine"])
#create label up-stable-down
# 0=increase, 1=decrease, 2=stable
if (dcf[i, "creatinine_tomorrow"] > dcf[i, "creatinine"] + 0.15)
  dcf[i, "label"] <- 0 #increase
else if ((dcf[i, "creatinine_tomorrow"] <= dcf[i, "creatinine"] + 0.15) & 
         (dcf[i, "creatinine_tomorrow"] >= dcf[i, "creatinine"] - 0.15))
  dcf[i, "label"] <- 2 #stable
else if (dcf[i, "creatinine_tomorrow"] < dcf[i, "creatinine"] - 0.15)
  dcf[i, "label"] <- 1 #decrease
}

#exclude patients without value tomorrow
dcf_small <- dcf[!is.na(dcf$creatinine_tomorrow) & (dcf$creatinine_tomorrow != -Inf) &
                   !is.na(dcf$label), ]
#exclude NA 
dcf_small <- dcf_small[!is.na(dcf_small$aki_7day) & (!is.na(dcf_small$aki_stage_7day)), ]
write.csv(dcf_small, "../results/tomorrows_value.csv")
