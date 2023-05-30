# Bamboo-Classification-using-AdaBoost-Ensemble-Technique-and-Voting-Algorithms
Boosting ensembles are commonly used to enhance the accuracy of classification models by training multiple models sequentially and merging their predictions. However, selecting the best model from the ensemble can be challenging. Voting algorithms are a popular method for selecting the best-fit model by aggregating the predictions of multiple models. However, existing voting algorithms have limitations, including hard voting being sensitive to class imbalance and soft voting assuming that all models are equally reliable. Weighted voting is also challenging, requiring expert knowledge and experience to assign weights to models.

In this project, firstly four boosting ensembles are trained using Naive Bayes (NB), Support Vector Machine (SVM), Classification and Regression Trees (CART), and Random Forest (RF) as base classifiers. Later, apply voting algorithms such as hard voting, soft voting, weighted voting, and others on these four ensembles. Accuracy, Precision, Recall, F1-Score and Confusion Matrix are taken as the metric for evaluating the performance. Dataset is collected from two bamboo-dominated districts named Dima Hasao and Karbi Anglong, one as a training site and the other as a validating site in Assam state.

## Dataset Preparation
The DIMA HASAO and KARBI ANGLONG dataset is used for this project. The data preparation begins with importing the district shapefile to define the region of interest. The Google Earth Code Editor map is then upgraded to include the district layer. Sentinel data from Copernicus.eu is imported to provide high-resolution satellite imagery. Manual annotation is performed to create training data for six classes: water, land, pure bamboo, mixed bamboo, dominant bamboo, and forest. The annotated data is converted to a CSV file format, containing coordinates and class labels. Classification algorithms are trained using this data for accurate classification of unseen satellite imagery within the districts.

After dataset is prepared we preprocess the data like removing duplicates and filling the missing values

## Model training and Results
By using AdaBoost Ensemble Technique we train our base models which are:SVM, Naive Bayes, Random Forest, CART. Once we obtain the four ensembles we apply the voting Algorithms(hard,soft,weighted) to obtain the final improved accuracy.
To evaluate the performance we used performance evaluation metrics:Accuracy, F1 score, Precision, Recall, Confusion matrix.

## Results
for hard-voting:
- Accuracy : 94.69696969697
- Precison : 94.98332123432
- Recall : 94.696969696967
- F1 score : 93.61803160567

for soft-voting:
- Accuracy : 97.727272727273
- Precison : 97.737431789765
- Recall : 97.7272727272723
- F1 score : 97.774556472

for weighted-voting:
- Accuracy : 89.393939393939
- Precison : 91.71122994674
- Recall : 89.39393939393939
- F1 score : 89.706945675742






