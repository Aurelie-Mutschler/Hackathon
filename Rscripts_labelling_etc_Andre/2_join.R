#### header ####
# This script is NOT a necessary part of the creat-forecast workflow. It is only a shortcut 
# for the labelling algorithm. Once the labelling is done on a sufficiently larg cohort, we can 
# label cohorts of the same or a smaller cohort (opt. wiht new feaures) by merging the old labels 
# to the new cohort. (keeping all entries from the new cohort and dropping those who do not match 
# from the old labeled cohort)
# 
# Input:
#    + ../data/tomorrows_value.csv : old cohort with labels
#    + ../results/creatinine_measurements.csv : new cohort from Aureli
#    + ../data/static_data.csv : static data with additional features
#
# Output:
#    + ../results/tomorrows_value_final.csv : final dataset
#
#
# Author: Andre Beinrucker TMO

# initialize workspace
rm(list=ls())
graphics.off()

#read data dl from Saturday (with labels) with Aurelies data da from Sunday morning (with features)
#and static data with additional features ds
da <- read.csv(file="../data/creatinine_measurements.csv", stringsAsFactors = FALSE)
dl <- read.csv(file="../results/tomorrows_value.csv", stringsAsFactors = FALSE)
ds <- read.csv(file="../data/static_data.csv", stringsAsFactors = FALSE)

#not essential: check that creatinine time is almost unique id
#table(table(da$creatinine_time))
#not essential: remove duplicated creatinine_times 
#daMultipleTimes <- da[duplicated(da$creatinine_time), ]

#create almost unique id  to merge later
da$mergeID <- paste(da$icustay_id, da$creatinine_time)
dl$mergeID <- paste(dl$icustay_id, dl$creatinine_time)

#check uniqueness of new id
table(table(da$mergeID))
table(table(dl$mergeID))

#remove samples with non-unique id (1 in Aurelies data), check again
da <- da[!duplicated(da$mergeID), ]
table(table(da$mergeID))

#merge data
#d <-merge(dl, da, by="mergeID", all=FALSE)
d <-merge(dl[,c("mergeID", "label", "creatinine_tomorrow", "creatinine_yesterday",
                "creatinine_before_yesterday"), ], da, by="mergeID", all=FALSE)

#make creatinine values numeric (remove -Inf)
d[d$creatinine_yesterday== -Inf, "creatinine_yesterday"] <- NA
d[d$creatinine_before_yesterday== -Inf, "creatinine_before_yesterday"] <- NA

#unique(d$creatinine_yesterday)
#remove column "X"
d <- d[,-which(names(d) %in% c("X"))]

#remove column "X"
d <- d[,-which(names(d) %in% c("X"))]

#only keep one line per icustay, as static information does not change within one stay
dsFiltered <- ds[!duplicated(ds$icustay_id), ]
dsFiltered <- dsFiltered[, c("icustay_id", "ethnicity", "diagnosis", "gender")]

#filter and join with static data
dFinal <- merge(d, dsFiltered, by="icustay_id", all.x=TRUE, all.y=FALSE)
write.csv(dFinal, "../results/tomorrows_value_final.csv")
