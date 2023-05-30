import pandas as pd
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import tkinter as tk
from tkinter import filedialog

def browse_file():
    file_path = filedialog.askopenfilename()
    path_entry.delete(0, tk.END)
    path_entry.insert(0, file_path)


def classify():
    file_path = path_entry.get()
    df = pd.read_csv(file_path)
    df['latitude'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'][1])
    df['longitude'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'][0])
    X = df[['latitude', 'longitude', 'system:index']]
    y = df['Class']

    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(), inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    svm = SVC(probability=True)
    nb = GaussianNB()
    rf = RandomForestClassifier()
    cart = DecisionTreeClassifier()

    svm_ensemble = AdaBoostClassifier(base_estimator=svm)
    nb_ensemble = AdaBoostClassifier(base_estimator=nb)
    rf_ensemble = AdaBoostClassifier(base_estimator=rf)
    cart_ensemble = AdaBoostClassifier(base_estimator=cart)
    
    voting_clf_hard = VotingClassifier(estimators=[('svm', svm_ensemble), ('nb', nb_ensemble), ('rf', rf_ensemble), ('cart', cart_ensemble)], voting='hard')
    voting_clf_soft = VotingClassifier(estimators=[('svm', svm_ensemble), ('nb', nb_ensemble), ('rf', rf_ensemble), ('cart', cart_ensemble)], voting='soft')
    weights = [0.9,0.6, 0.3, 0.5]
    voting_clf_weighted = VotingClassifier(estimators=[('svm', svm_ensemble), ('nb', nb_ensemble), ('rf', rf_ensemble), ('cart', cart_ensemble)], voting='soft', weights=weights)
    
    svm_ensemble.fit(X_train, y_train)
    nb_ensemble.fit(X_train, y_train)
    rf_ensemble.fit(X_train, y_train)
    cart_ensemble.fit(X_train, y_train)

    voting_clf_hard.fit(X_train, y_train)
    voting_clf_soft.fit(X_train, y_train)
    voting_clf_weighted.fit(X_train, y_train)

    svm_y_pred = svm_ensemble.predict(X_test)
    nb_y_pred = nb_ensemble.predict(X_test)
    rf_y_pred = rf_ensemble.predict(X_test)
    cart_y_pred = cart_ensemble.predict(X_test)

    voting_clf_hard_y_pred = voting_clf_hard.predict(X_test)
    voting_clf_soft_y_pred = voting_clf_soft.predict(X_test)
    voting_clf_weighted_y_pred = voting_clf_weighted.predict(X_test)

    svm_accuracy = accuracy_score(y_test, svm_y_pred)
    nb_accuracy = accuracy_score(y_test, nb_y_pred)
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    cart_accuracy = accuracy_score(y_test, cart_y_pred)

    voting_clf_hard_accuracy = accuracy_score(y_test, voting_clf_hard_y_pred)
    voting_clf_soft_accuracy = accuracy_score(y_test, voting_clf_soft_y_pred)
    voting_clf_weighted_accuracy = accuracy_score(y_test, voting_clf_weighted_y_pred)

    output_text.delete(1.0, tk.END) 
    output_text.insert(tk.END, f"SVM accuracy: {svm_accuracy*100:.2f}\n")
    output_text.insert(tk.END, f"Naive Bayes accuracy: {nb_accuracy*100:.2f}\n")
    output_text.insert(tk.END, f"Random Forest accuracy: {rf_accuracy*100:.2f}\n")
    output_text.insert(tk.END, f"Decision Tree accuracy: {cart_accuracy*100:.2f}\n")
    output_text.insert(tk.END, f"Voting classifier (hard) accuracy: {voting_clf_hard_accuracy*100:.2f}\n")
    output_text.insert(tk.END, f"Voting classifier (soft) accuracy: {voting_clf_soft_accuracy*100:.2f}\n")
    output_text.insert(tk.END, f"Weighted voting classifier (soft) accuracy: {voting_clf_weighted_accuracy*100:.2f}\n")


root = tk.Tk()
root.geometry("600x400")
root.title("Bamboo Classification")

path_label = tk.Label(root, text="Select CSV file:")
path_label.pack()
path_entry = tk.Entry(root, width=50)
path_entry.pack()
browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.pack()

classify_label = tk.Label(root, text="Click button to classify:")
classify_label.pack()
classify_button = tk.Button(root, text="Classify", command=classify)
classify_button.pack()
output_text = tk.Text(root)
output_text.pack()

root.mainloop()







