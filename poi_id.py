#!/usr/bin/python
# importer functions
import sys
import pickle
import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from collections import OrderedDict

### HELPER functions
# functions that help to clean and format the code
def dump_classifier_and_data(clf, dataset, feature_list):
    CLF_PICKLE_FILENAME = "my_classifier.pkl"
    DATASET_PICKLE_FILENAME = "my_dataset.pkl"
    FEATURE_LIST_FILENAME = "my_feature_list.pkl"

    with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 3% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    mypred = [p[0] for p in predictions]
    myages = [p[0] for p in ages]
    myNW = [p[0] for p in net_worths]
    err = []
    err_sq = []
    for i in range(len(mypred)):
        error = myNW[i] - mypred[i]
        error_sq = error * error
        err.append(error)
        err_sq.append(error_sq)
    master_list = zip(err_sq,myages,myNW,err)

    sort_master_list = sorted(master_list, key=lambda t: t[0])

    myList = sort_master_list[: int(round(0.97 * len(sort_master_list)))]
    cleaned_data = [(t[1], t[2], t[3]) for t in myList]
    return cleaned_data

"""
    A general tool for converting data from the
    dictionary format to an (n x k) python list that's
    ready for training an sklearn algorithm

    n--no. of key-value pairs in dictonary
    k--no. of features being extracted

    dictionary keys are names of persons in dataset
    dictionary values are dictionaries, where each
        key-value pair in the dict is the name
        of a feature, and its value for that person

    In addition to converting a dictionary to a numpy
    array, you may want to separate the labels from the
    features--this is what targetFeatureSplit is for

    so, if you want to have the poi label as the target,
    and the features you want to use are the person's
    salary and bonus, here's what you would do:

    feature_list = ["poi", "salary", "bonus"]
    data_array = featureFormat( data_dictionary, feature_list )
    label, features = targetFeatureSplit(data_array)

    the line above (targetFeatureSplit) assumes that the
    label is the _first_ item in feature_list--very important
    that poi is listed first!
"""

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """


    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()
        #print(keys)
    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print( "error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)

def targetFeatureSplit( data ):
    """
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as
        input formats when training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#this is the initial list of data to explore (not final!)
features_list = ['poi','salary','bonus','exercised_stock_options','from_messages','from_poi_to_this_person',
'from_this_person_to_poi','deferred_income'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r",) as data_file:
    data_dict = pickle.load(data_file)
    # Take out total option (from outlier section below)
    data_dict.pop('TOTAL')

# Create lists for use in basic analysis
names, Salary, Bonus, ESO, FEmails, FromPOI, POI, DefInc = [], [], [], [], [], [], [], []
count_nan, count_all = 0, 0

#counting the total data points, POI and number of NaNs
for person, data in data_dict.iteritems():
    names.append(person)
    for key, value in data.iteritems():
        count_all += 1
        if value == 'NaN':
            count_nan = count_nan + 1
for name in names:
    POI.append(int(data_dict[name]['poi']))
    Salary.append(float(data_dict[name]['salary']))
    Bonus.append(float(data_dict[name]['bonus']))
    ESO.append(float(data_dict[name]['exercised_stock_options']))
    DefInc.append(float(data_dict[name]['deferred_income']))
    FromPOI.append(float(data_dict[name]['from_poi_to_this_person']))
    FEmails.append(float(data_dict[name]['from_messages']))

print 'Total People: ' + str(len(names))
print 'Total POI: ' + str(sum(POI))
print 'Total Data Points: ' + str(count_all)
print 'Total NaN: ' + str(count_nan)

### Task 2: Remove outliers
# first look at the data we have and where the POIs are to see if there
#are any obvious outliers (for eg TOTAL)
plt.scatter(Salary, Bonus, color='green')
for i, value in enumerate(POI):
    if POI[i] ==1:
        plt.scatter(Salary[i], Bonus[i], color = 'red')
plt.show()
plt.scatter(FEmails, FromPOI, color='green')
for i, value in enumerate(POI):
    if POI[i] ==1:
        plt.scatter(FEmails[i], FromPOI[i], color = 'red')
plt.show()
#shape data for linear regression model for testing if removing outliers is effective
eso = np.nan_to_num(np.reshape((np.array(ESO)), (len(ESO), 1)))
poi = np.nan_to_num(np.reshape((np.array(Salary)), (len(Salary), 1)))

eso_train, eso_test, poi_train, poi_test = train_test_split(eso, poi, test_size=0.2, random_state=15)
reg = linear_model.LinearRegression()
reg.fit(eso_train, poi_train)
print('Regression Score Pre-Clean: ' + str(reg.score(eso_test, poi_test)))

try:
    plt.plot(eso, reg.predict(eso), color="blue")
except NameError:
    pass
plt.scatter(eso, poi)
plt.show()

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(eso_train)
    cleaned_data = outlierCleaner( predictions, eso_train, poi_train )
except NameError:
    print( "your regression object doesn't exist, or isn't name reg")
    print( "can't make predictions to use in identifying outliers")

### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    eso, poi, errors = zip(*cleaned_data)
    eso       = np.reshape( np.array(eso), (len(eso), 1))
    poi = np.reshape( np.array(poi), (len(poi), 1))
    ### refit your cleaned data!
    try:
        reg.fit(eso, poi)
        print 'Regression Score Post Clean: ' + str(reg.score(eso_test, poi_test))
        plt.plot(eso, reg.predict(eso), color="blue")
    except NameError:
        print( "you don't seem to have regression imported/created")
        print( "   or else your regression object isn't named reg")
        print( "   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(eso, poi)
    plt.xlabel("eso")
    plt.ylabel("poi")
    #plt.show()
else:
    print( "outlierCleaner() is returning an empty list, no refitting to be done")

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# create np arrays for the rescaler
salary       = np.nan_to_num(np.reshape((np.array(Salary)), (len(Salary), 1)))
bonus       = np.nan_to_num(np.reshape((np.array(Bonus)), (len(Bonus), 1)))
eso       = np.nan_to_num(np.reshape((np.array(ESO)), (len(ESO), 1)))
#rescale salary, bonus and exercised sti=ock options
scaler = MinMaxScaler()
scaled = scaler.fit(salary)
scaled_salary = scaled.transform(salary)
scaled = scaler.fit(bonus)
scaled_bonus = scaled.transform(bonus)
scaled = scaler.fit(eso)
scaled_eso = scaled.transform(eso)
#append data_dict with rescaled values and create new
#value for the percent of emails received from POI
count = 0
for name in names:
    data_dict[name]['salary'] = scaled_salary[count][0]
    data_dict[name]['bonus'] = scaled_bonus[count][0]
    data_dict[name]['exercised_stock_options'] = scaled_eso[count][0]
    count = count + 1
    data_dict[name]['Percent_Emails_from_POI'] = float(data_dict[name]['from_poi_to_this_person'])/float(data_dict[name]['from_messages'])

features = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
'director_fees','to_messages',  'from_poi_to_this_person', 'from_messages',
'from_this_person_to_poi', 'shared_receipt_with_poi']
data = featureFormat(data_dict, features)
# seperating the poi from rest of data
features_list = []
poi = []
for point in data:
    poi.append(point[0])
    features_list.append(point[1:])
# creating a split data set for use in selector - using train_test_split for simplicity
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features_list, poi, test_size=0.3, random_state=42)
# Run selector to get a score of effect for each feature
selector = SelectKBest(f_classif, k=5)
selector.fit(features_train, labels_train)
features_train = selector.transform(features_train)
features_test  = selector.transform(features_test)
# create a dictionary of the top scores
scores = selector.scores_
scores_dict = {}
scores_top_dict = {}
count = 0
for_using_features = []
for feature in features:
    if feature == 'poi':
        pass
    else:
        scores_dict[feature]=scores[count]
        if scores[count] > 2:
            scores_top_dict[feature] = scores[count]
            for_using_features.append(feature)
        count = count + 1
print(scores_top_dict)
# create new dataset with only the selected top features
new_data_dict = OrderedDict()
for person in data_dict:
    names.append(person)

for name in names:
    new_data_dict[name]=OrderedDict()
    new_data_dict[name]['poi']=data_dict[name]['poi']
    for feat in for_using_features:
        new_data_dict[name][feat] = data_dict[name][feat]
my_dataset = new_data_dict

new_features_list = ['poi']
for feat in for_using_features:
    new_features_list.append(feat)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.4, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#trying decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
recall=recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)
print 'TREE RECALL:' + str(recall)
print 'TREE PRECISION:' + str(prec)
print 'DECISION TREE ACCURACY: ' + str(accuracy)

#trying random forest
clf = RandomForestClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
recall=recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)
print 'RANDOM FOREST RECALL:' + str(recall)
print 'RANDOM FOREST PRECISION:' + str(prec)
print 'RANDOM FOREST ACCRAUCY:' + str(accuracy)

#trying SVM
clf = SVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
recall=recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)
print 'SVM RECALL:' + str(recall)
print 'SVM PRECISION:' + str(prec)
print 'SVM ACCURACY: '+str(accuracy)
# Trying Naive Bayes
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
recall=recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)
print 'Naive Bayes RECALL:' + str(recall)
print 'Naive Bayes PRECISION:' + str(prec)
print 'Naive Bayes ACCURACY: '+str(accuracy)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function.

# I spent some time trying to tune the Random Forest Classifier before settling
#on the Guassian Naive Bayes so I have left the code here as an example of returning
#given the Guassian Naive Bayes did not require it
# Iteration 1 - I ran the code multiple times varying the n_estimators
clf = RandomForestClassifier(n_estimators=200)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
recall=recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)
print 'RANDOM FOREST improved ACCRAUCY:' + str(accuracy)
print 'RANDOM FOREST improved RECALL:' + str(recall)
print 'RANDOM FOREST improved PRECISION:' + str(prec)
# Iteration 2
clf = RandomForestClassifier(criterion = 'entropy')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
recall=recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)
print 'RANDOM FOREST improved ACCRAUCY:' + str(accuracy)
print 'RANDOM FOREST improved RECALL:' + str(recall)
print 'RANDOM FOREST improved PRECISION:' + str(prec)
# Iteration 3 - I ran the code multiple times varying the mmax_depth
clf = RandomForestClassifier(max_depth = 20)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
recall=recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)
print 'RANDOM FOREST improved ACCRAUCY:' + str(accuracy)
print 'RANDOM FOREST improved RECALL:' + str(recall)
print 'RANDOM FOREST improved PRECISION:' + str(prec)
# Iteration 4 - I ran the code multiple times varying the min_samples_split
clf = RandomForestClassifier(min_samples_split = 4)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
recall=recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)
print 'RANDOM FOREST improved ACCRAUCY:' + str(accuracy)
print 'RANDOM FOREST improved RECALL:' + str(recall)
print 'RANDOM FOREST improved PRECISION:' + str(prec)
# Iteration 5 - I ran the code multiple times with different combinations of the above parameters
clf = RandomForestClassifier(n_estimators = 1000, oob_score = True, random_state = 30)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
recall=recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)
print 'RANDOM FOREST improved ACCRAUCY:' + str(accuracy)
print 'RANDOM FOREST RECALL:' + str(recall)
print 'RANDOM FOREST PRECISION:' + str(prec)

# Validation with Startified Shuffle Split
features_train, features_test, labels_train, labels_test = [],[],[],[]
sss = StratifiedShuffleSplit(n_splits = 3, test_size = 0.5, random_state = 15)
for train_indices, test_indices in sss.split(features, labels):
    for i in train_indices:
        features_train.append(features[i])
        labels_train.append(labels[i])
    for i in test_indices:
        features_test.append(features[i])
        labels_test.append(labels[i])

# final algorithm!
clf2 = GaussianNB()
clf2.fit(features_train, labels_train)
pred = clf2.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
recall=recall_score(labels_test, pred)
prec = precision_score(labels_test, pred)

print 'GNB improved SSS ACCRAUCY:' + str(accuracy)
print 'GNB improved SSS RECALL:' + str(recall)
print 'GNB improved SSS PRECISION:' + str(prec)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf2, my_dataset, new_features_list)
