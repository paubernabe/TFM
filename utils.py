#custom utils

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



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


def plot_class_distribution(X, y):

    #reduce data dimensionality using another PCA
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