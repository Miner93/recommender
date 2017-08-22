from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



appended_df = []

for user in range(n_users):
    ratings = pd.DataFrame(data=R.loc[user,R.loc[user,:] > 0])
    ratings['Prediction'] = R_hat.loc[user,R.loc[user,:] > 0]
    ratings.reset_index(level=0, inplace=True)
    ratings.columns = ['item_id', 'Actual Rating', 'Predicted Rating']
    ratings['Actual goodenough'] = np.where(ratings['Actual Rating'] >= 4.0, 1, 0)
    appended_df.append(ratings)
    
appended_df = pd.concat(appended_df, ignore_index=True)

appended_df = appended_df.sort_values(by='Predicted Rating', ascending=False)


# ROC

actual = np.array(appended_df['Actual goodenough'])
predicted = np.array(appended_df['Predicted Rating'])

fpr, tpr, _ = roc_curve(actual, predicted, pos_label=1)
roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='Krzywa ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc="lower right")
plt.show()



#Precision-Recall

precision, recall, _ = precision_recall_curve(actual, predicted, pos_label=1)
average_precision = average_precision_score(actual, predicted)


plt.clf()
plt.plot(recall, precision, lw=lw, color='navy',
         label='Krzywa Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower right")
plt.show()


#RMSE - ALS

plt.plot(range(n_epochs), train_errors, marker='o', label='Zbiór treningowy');
plt.plot(range(n_epochs), test_errors, marker='v', label='Zbiór testowy');
plt.xlabel('Liczba iteracji');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()


#RMSE - kNN

appended_df = []

for user in range(n_users):
    ratings = pd.DataFrame(data=A.loc[user,A.loc[user,:] > 0])
    ratings['Prediction'] = T_hat.loc[user,A.loc[user,:] > 0]
    ratings.reset_index(level=0, inplace=True)
    ratings.columns = ['item_id', 'Actual Rating', 'Predicted Rating']
    appended_df.append(ratings)
    
appended_df = pd.concat(appended_df, ignore_index=True)

appended_df['(Actual-Predicted)^2'] = (appended_df['Actual Rating'] - appended_df['Predicted Rating'])**2
total = np.sum(appended_df['(Actual-Predicted)^2'])
n = appended_df.shape[0]
rmse = np.sqrt(total / n)
