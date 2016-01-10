import csv as csv
import numpy as np 

csv_file_object = csv.reader(open('train.csv', 'rb')) 
header = csv_file_object.next()
data=[] 
for row in csv_file_object:
	data.append(row)
data = np.array(data) 

# print data[0]
# print data[-1]
# print data[:,4]

number_passengers = np.size(data[:,1].astype(np.float))
number_survived = np.sum(data[:,1].astype(np.float))
proportion_survivors = number_survived/number_passengers

women_only_stats = data[:,4] == "female"
men_only_stats = data[:,4] != "female"

women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

proportions_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportions_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print "Proportion of women who survived is %s" % proportions_women_survived
print "Proportion of men who survived is %s" % proportions_men_survived

