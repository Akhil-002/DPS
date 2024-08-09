from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from collections import Counter
import csv
import numpy as np
import pandas as pd
import os
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup


data = pd.read_csv(os.path.join("templates", "Training.csv"))
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

# print(y)

print("DecisionTree")
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

y_pred_rf = rf.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

y_pred = nb_classifier.predict(x_test)
accuracy_nb = accuracy_score(y_test, y_pred)

svm_predictions = svm_classifier.predict(x_test)
accuracy_svm = accuracy_score(y_test, svm_predictions)

log_reg_classifier = LogisticRegression(random_state=42)
log_reg_classifier.fit(x_train, y_train)
y_pred_log_reg = log_reg_classifier.predict(x_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

print(accuracy_dt,accuracy_nb,accuracy_rf)





indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]

dictionary = dict(zip(symptoms, indices))

def get_info(disease):
    url = f'https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term='+disease[0].replace(' ','+')
    response = requests.get(url)
    xml_response = response.text

    root = ET.fromstring(xml_response)

    # Extract the content within the FullSummary tag
    data = root.find('.//content[@name="FullSummary"]').text
    souce = root.find('.//content[@name="organizationName"]').text
    soup = BeautifulSoup(data, 'html.parser')
    cleaned_text = soup.get_text(separator=' ')

    return cleaned_text,souce

def majority_prediction(my_list):

    if not my_list:  # Handle empty list gracefully
        return [], 0

    # Create a dictionary to store counts of unique strings
    string_counts = {}
    for item in my_list:
        string_counts[item] = string_counts.get(item, 0) + 1

    # Find the maximum count
    max_count = max(string_counts.values())

    # Get all strings with the maximum count
    most_frequent_strings = [item for item, count in string_counts.items()
                             if count == max_count]

    return most_frequent_strings

def dosomething(symptom):
    print(symptom)
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1, 1)).transpose()
    
  
    disease_dt = dt.predict(user_input_label)
    disease_rf = rf.predict(user_input_label)
    disease_nb = nb_classifier.predict(user_input_label)
    disease_svm = svm_classifier.predict(user_input_label)
    disease_lg = log_reg_classifier.predict(user_input_label)

    disease = majority_prediction([disease_rf[0],disease_dt[0],disease_nb[0],disease_svm[0],disease_lg[0]])
    info,src = get_info(disease)

    dic = { 'Naive Bayes':[disease_nb,accuracy_nb*100],
            'Decision Tree':[disease_dt,accuracy_dt*100],
            'Random Forest':[disease_rf,accuracy_rf*100],
            'SVM':[disease_svm,accuracy_svm*100],
            'Logistic Regression':[disease_lg,accuracy_log_reg*100]
          }

    return dic,info,src,disease,symptom


# print(prediction)
#   input_feature_indices = []
#     input_feature_importances = []
#     for feature in user_input_symptoms:
#         index = user_input_symptoms.index(feature)
#         importance = feature_importances[index]
#         input_feature_indices.append(index)
#         input_feature_importances.append(importance)

#     total_importance_input_features = sum(input_feature_importances)


#     feature_importances_percentage_input_features = [
#         importance * 100 / total_importance_input_features for importance in input_feature_importances]

#     result_dict = {}

#     for index, importance_percentage in zip(input_feature_indices, feature_importances_percentage_input_features):
#         symptom = user_input_symptoms[index]
#         result_dict[symptom] = importance_percentage
#     sorted_result_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}
