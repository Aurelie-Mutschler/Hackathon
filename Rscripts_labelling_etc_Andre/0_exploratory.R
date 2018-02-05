#### header ####
# This script is for visualistion, NOT a necessary part of the creat-forecase workflow. It creates a cohort from the MIMIC database, following the tutorials in:
# https://github.com/MIT-LCP/mimic-code/blob/master/tutorials/using_r_with_jupyter.ipynb
# and
# https://github.com/MIT-LCP/mimic-code/blob/master/tutorials/cohort-selection.ipynb
# adds creatinine values and visualises the dynamics of reatinine for the first icustays
# 
# Input:
#    + MIMIC database
#
# Output:
#    + pdf with dynamic of creatinine
#
#
# Author: Andre Beinrucker TMO

#### initialisation ####
#adapt working directory here
setwd("C:/Users/andre.beinrucker/Downloads/MIMIC/programs")
#if you do not have checkpoint installed, you need to run: 
#install.packages(checkpoint)
library(checkpoint)
#all further packages will be installed by checkpoint, set the location:
#checkpoint("2017-08-14", checkpointLocation = "I:/Bereichsaustausch/Biostatistics/Software/StatisticsTools/R/checkpoint_library", R.version="3.1.2")
checkpoint("2017-08-14", checkpointLocation = "C:/Users/andre.beinrucker/Downloads/checkpoint_library", R.version="3.1.2")
require("RPostgreSQL")
require("ggplot2")

# load the PostgreSQL driver
drv <- dbDriver("PostgreSQL")

# create a connection to the postgres database
# set the search path to the mimiciii schema
con <- dbConnect(drv, dbname = "mimic",
                 host = "localhost", port = 5432,
                 user = "postgres")
#this gives an error, I do not know why
dbSendQuery(con, 'set search_path to mimiciii')

#### select cohort ####
cohort = dbGetQuery(con,"
WITH co AS
(
  SELECT icu.subject_id, icu.hadm_id, icu.icustay_id
  , EXTRACT(EPOCH FROM outtime - intime)/60.0/60.0/24.0 as icu_length_of_stay
  , EXTRACT('epoch' from icu.intime - pat.dob) / 60.0 / 60.0 / 24.0 / 365.242 as age
  , RANK() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS icustay_id_order
  FROM icustays icu
  INNER JOIN patients pat
  ON icu.subject_id = pat.subject_id
  LIMIT 100000
)
                     , serv AS
                     (
                     SELECT icu.hadm_id, icu.icustay_id, se.curr_service
                     , CASE
                     WHEN curr_service like '%SURG' then 1
                     WHEN curr_service = 'ORTHO' then 1
                     ELSE 0 END
                     as surgical
                     , RANK() OVER (PARTITION BY icu.hadm_id ORDER BY se.transfertime DESC) as rank
                     FROM icustays icu
                     LEFT JOIN services se
                     ON icu.hadm_id = se.hadm_id
                     AND se.transfertime < icu.intime + interval '12' hour
                     )
                     SELECT
                     co.subject_id, co.hadm_id, co.icustay_id, co.icu_length_of_stay
                     , co.age
                     , co.icustay_id_order
                     
                     , CASE
                     WHEN co.icu_length_of_stay < 2 then 1
                     ELSE 0 END
                     AS exclusion_los
                     , CASE
                     WHEN co.age < 16 then 1
                     ELSE 0 END
                     AS exclusion_age
                     , CASE 
                     WHEN co.icustay_id_order != 1 THEN 1
                     ELSE 0 END 
                     AS exclusion_first_stay
                     , CASE
                     WHEN serv.surgical = 1 THEN 1
                     ELSE 0 END
                     as exclusion_surgical
                     FROM co
                     LEFT JOIN serv
                     ON  co.icustay_id = serv.icustay_id
                     AND serv.rank = 1
")
print(head(cohort))
cohort$excluded <- cohort$exclusion_age + cohort$exclusion_first_stay + cohort$exclusion_los + cohort$exclusion_surgical > 0
#how many are not excluded?
sum(1-cohort$excluded)
#create cohort without exclusions
cohortFiltered <- cohort[cohort$excluded ==0,]

#### creatinine table ####
creatinineTable = dbGetQuery(con, 
"SELECT subject_id, valuenum, charttime
FROM labevents
WHERE itemid = 50912
LIMIT 800000")
print(head(creatinineTable))
length(unique(creatinineTable$subject_id))

#merge cohortFiltered with creatinineTable
d <- merge(cohortFiltered, creatinineTable, by="subject_id", all = FALSE)
print(head(d))
length(unique(d$subject_id))
#sort data frame
#d <- d[order(d$subject_id, d$charttime),]
d <- d[order(d$subject_id, d$charttime),]

#plot patient creatinine values
pdf("../results/creatinineFrequencyAndDistribution.pdf")
# histogram of number of creatinine values
plot(table(table(d$subject_id)), xlab="number of creatinine values of one pat.",
     ylab="frequency (number of pat. in dataset)", main="How many creatinine values per patient do we have?")
## histogram of number of creatinine values, zoomed
plot(table(table(d$subject_id)), xlab="number of creatinine values of one pat.",
     ylab="frequency (number of pat. in dataset)", xlim=c(0, 50), main="How many creatinine values per patient do we have? (Zoomed)")
#boxplots of creatinine values
boxplot(d$valuenum, main="Boxplot of all Creatinine Values", ylab="creatinine mg/dL")
#boxplots of creatinine values, zoomed
boxplot(d$valuenum, main="Boxplot zoomed till 10", ylab="creatinine mg/dL", ylim=c(0,10))
dev.off()

pdf("../results/creatinineOverTime.pdf")
#get max creatinine for plot
ymax <- 0
nbPatientsToPlot <- 100
for (pat in unique(d$subject_id)[1:nbPatientsToPlot])
{
   ymax <- max(ymax, max(d[d$subject_id==pat, "valuenum"]))
}
#plot evolution for each patient
for (pat in unique(d$subject_id)[1:nbPatientsToPlot])
{
  firstDate <- min(d[d$subject_id==pat, "charttime"])
plot(
     difftime(d[d$subject_id==pat, "charttime"]
              [1:min(10, length(d[d$subject_id==pat, "charttime"]))], 
              firstDate, unit="days"), 
     d[d$subject_id==pat, "valuenum"]
      [1:min(10, length(d[d$subject_id==pat, "charttime"]))],
    xlab="days after first creatinine draw",
    ylab="creatinine value mg/dL",
    ylim=c(0.1,ymax),
    log="y"
         )
lines(difftime(d[d$subject_id==pat, "charttime"]
               [1:min(10, length(d[d$subject_id==pat, "charttime"]))], 
               firstDate, unit="days"),
      d[d$subject_id==pat, "valuenum"]
      [1:min(10, length(d[d$subject_id==pat, "charttime"]))],
      xlab="days after first creatinine draw",
      ylab="creatinine value mg/dL")
}
dev.off()

median(d$valuenum, na.rm=TRUE)

#### close the connection ####
dbDisconnect(con)
dbUnloadDriver(drv)