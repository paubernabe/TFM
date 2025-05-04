#custom utils

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

'''
Compute metrics
'''
def classification_metrics(y_true, y_pred, model_name, experiment_name, results_df):
    
    
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    df_per_class = pd.DataFrame({
        'Model': model_name,
        'Data Type': experiment_name,
        'Class': sorted(set(y_true)),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Macro F1': macro_f1,
        'Weighted F1': weighted_f1
    })
    df_per_class = df_per_class.round(3)

    results_df.append(df_per_class[df_per_class['Class'] == 1])


    return df_per_class

"""
plots the roc curve based of the probabilities
"""
def plot_roc_curve(true_y, y_prob, model_name="model"):
    
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print(model_name, ' has an AUC score of ', roc_auc_score(true_y, y_prob))
##################

'''
Plot precision recall
'''
def plot_precision_recall(y_true, y_probs):
  precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

  # Calculate the Average Precision (AP) score
  average_precision = average_precision_score(y_true, y_probs)

  # Plot the Precision-Recall curve
  plt.figure(figsize=(10, 6))
  plt.plot(recall, precision, color='b', label=f'Precision-Recall curve (AP={average_precision:.2f})')
  plt.fill_between(recall, precision, color='lightblue', alpha=0.5)

  # Titles and labels
  plt.title('Precision-Recall Curve', fontsize=16)
  plt.xlabel('Recall', fontsize=12)
  plt.ylabel('Precision', fontsize=12)
  plt.legend(loc='best')
  plt.grid(True)
  plt.show()


'''
Reduces data dimensionality to be plotted and see the class distribution
'''
def plot_class_distribution(X, y):

  X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)

  plt.figure(figsize=(8, 6))
  scatter = plt.scatter(
      X_reduced_pca[:, 0], X_reduced_pca[:, 1],
      c=y, cmap='coolwarm', edgecolors='k', linewidths=1, alpha=0.8
  )

  plt.title('PCA - Fraud Detection')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  
  # Custom legend
  handles = [
      plt.Line2D([0], [0], marker='o', color='w', label='No Fraud', markerfacecolor='blue', markersize=10, markeredgecolor='k'),
      plt.Line2D([0], [0], marker='o', color='w', label='Fraud', markerfacecolor='red', markersize=10, markeredgecolor='k')
  ]
  plt.legend(handles=handles)
  
  plt.grid(True)
  plt.tight_layout()
  plt.show()

'''
################PLOT CONFUSION MATRIX################################################################
'''

def plot_confusion_matrix(y_true, y_pred):
    # y_test: true labels
    # y_pred: predicted labels (0 or 1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    labels = ["Not Fraud", "Fraud"]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title("Confusion Matrix: Fraud Detection")
    plt.show()

'''
#################################################################################
'''

'''
################CROSS VALIDATION################################################################
'''
def grid_search_best_model(model, param_grid, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
    'recall': 'recall',
    'accuracy': 'accuracy'
    }

    grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           cv=skf,
                           scoring=scoring,
                            refit='accuracy',
                           n_jobs=-1,
                           verbose=1)
    
    # Fit
    grid_search.fit(X, y)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    ##
    return grid_search.best_estimator_
    


'''
#################################################################################################
'''

  