rm(list = ls())

library(seismic)

library(lattice)
library(ggplot2)
library(caret) # for recall, precision, f1 calculation
library (ROCR) # for auc calculation

data_ <- read.csv("weibo_cascades_processed.csv")

num_cascades = max(data_[, 2])
len_data = length(data_[, 2])
burst_min_len = 800
non_burst_max_len = 600
Observation_len = 6 * 60 * 60 
time_bin = 7200
prediction_length = 60 * 24 * 60 * 60
k = 50

burst_cascades = list()
non_burst_cascades = list()
burst_cascades_counter = 1
non_burst_cascades_counter = 1

print(num_cascades)

start_index = 0
total_cascades = 0
end = 0
for (i in c(1:len_data)) {
  curr_index = data_[i, 2]
  print(curr_index)
  if (curr_index != start_index){
    begin = end + 1
    end = i - 1
    start_index = curr_index
    total_cascades = total_cascades + 1
  }else{
    next
  }
  cascade = data_[begin:end,c(1,3)]
  min_cascade = min(cascade[, 1])
  cascade$vote_date = cascade$vote_date - min_cascade
  #cascade = sort(cascade)
  cascade = cascade[order(cascade$vote_date), ]
  cascade_total_retweets = length(cascade$vote_date)
  if (cascade_total_retweets > burst_min_len) {
    burst_cascades[[burst_cascades_counter]] = cascade
    burst_cascades_counter = burst_cascades_counter + 1
    if (burst_cascades_counter > 1000 && non_burst_cascades_counter > 1000){
      break
    }
  }
  
  if (cascade_total_retweets < non_burst_max_len) {
    non_burst_cascades[[non_burst_cascades_counter]] = cascade
    non_burst_cascades_counter = non_burst_cascades_counter + 1
  }
}

print(length(non_burst_cascades))
print(length(burst_cascades))

selected_non_burst_cascades = non_burst_cascades[1:length(burst_cascades)]
selected_cascades = c(burst_cascades, selected_non_burst_cascades)
labels = c(rep(1, length(burst_cascades)), rep(0, length(burst_cascades)))
predictions = c()
pred.time <- seq(0, prediction_length, by = time_bin)
print(pred.time)
cascade_sizes = c()
for (i in c(1:length(selected_cascades))) {
  print(i)
  tweet = selected_cascades[[i]]
  cascade_times <- tweet$vote_date
  followers <- tweet$follower_count
  followers = followers[cascade_times < Observation_len]
  cascade_times = cascade_times[cascade_times < Observation_len]
  infectiousness <- get.infectiousness(cascade_times, followers, pred.time)
  pred <- pred.cascade(pred.time, infectiousness$infectiousness, cascade_times, followers, n.star = 231.33)
  last_pred = mean(tail(pred, n=5))
  if (last_pred > burst_min_len) {
    predictions = c(predictions, 1)
  }
  else {
    predictions = c(predictions, 0)
  }
  cascade_sizes = c(cascade_sizes, length(tweet$vote_date))
}

#j=3
#pred.time <- seq(0, 60 * 24 * 60 * 60, by = 7200)
#obs =  2 * 24 * 60 * 60
#tweet = selected_cascades[[j]]
#cascade_times <- tweet$vote_date
#plot(cascade_times)
#print(length(cascade_times))
#followers <- tweet$follower_count
#followers = followers[cascade_times < obs]
#cascade_times = cascade_times[cascade_times < obs]
#print(length(cascade_times))
#print(length(followers))
#infectiousness <- get.infectiousness(cascade_times, followers, pred.time)
#pred <- pred.cascade(pred.time, infectiousness$infectiousness, cascade_times, followers, n.star = 231.33)
#plot(pred.time, infectiousness$infectiousness)
#plot(pred.time, pred)
#plot(tweet$vote_date)
#print(mean(tail(pred)))

acc = 0
for (i in c(1:length(labels))) {
  if (labels[i] == predictions[i]) {
    acc = acc + 1
  }
}

acc = acc / length(labels)
print(acc)

predicted <- factor(as.character(predictions), levels=unique(as.character(labels)))
expected  <- as.factor(labels)

burst_precision = posPredValue(predicted, expected, positive="1")
non_burst_precision = posPredValue(predicted, expected, positive="0")
burst_recall = sensitivity(predicted, expected, positive="1")
non_burst_recall = sensitivity(predicted, expected, positive="0")
burst_F1 = (2 * burst_precision * burst_recall) / (burst_precision + burst_recall)
non_burst_F1 = (2 * non_burst_precision * non_burst_recall) / (non_burst_precision + non_burst_recall)
print(burst_precision)
print(non_burst_precision)
print(burst_recall)
print(non_burst_recall)
print(burst_F1)
print(non_burst_F1)

pred_ <- prediction(predictions, labels)
auc.tmp <- performance(pred_,"auc")
auc <- as.numeric(auc.tmp@y.values)
print(auc)

sorted_idx <- order(cascade_sizes, decreasing = TRUE)
y_true <- labels[sorted_idx]
y_score <- predictions[sorted_idx]
y_true <- y_true[1:k]
y_score <- y_score[1:k]
print(sum(y_true * y_score) / sum(y_true))

# print(which(pred==max(pred)))
# print(which(cascade[,2]==max(cascade[,2])))
