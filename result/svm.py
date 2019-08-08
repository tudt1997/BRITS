# import xgboost as xgb
import numpy as np

model_name = 'brits'

impute = np.load('./{}_data.npy'.format(model_name)).reshape(-1, 48 * 35)
label = np.load('./{}_label.npy'.format(model_name))

data = np.nan_to_num(impute)

n_train = 3000

print(impute.shape)
print(label.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC


# data = StandardScaler().fit_transform(data)

Cs = [0.01, 0.1, 1.0, 10.0]
for C in Cs:
    model = LinearSVC(C=C, max_iter=10000, tol=1e-10).fit(data[:n_train], label[:n_train].ravel())
    pred = model.decision_function(data[n_train:]) #data[n_train:])
    auc = roc_auc_score(label[n_train:].ravel(), pred.ravel())
    print(C, auc)

