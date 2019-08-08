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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

auc = []

data = StandardScaler().fit_transform(data)

for i in range(5):
    #model =  RandomForestClassifier().fit(data[:n_train], label[:n_train])

    model = LinearSVC(max_iter=10000, tol=1e-10).fit(data[:n_train], label[:n_train].ravel())
    pred = model.predict(data[n_train:])
    #auc.append(roc_auc_score(label[n_train:].reshape(-1,), pred[:, 1].reshape(-1, )))
    auc.append(roc_auc_score(label[n_train:].ravel(), pred.ravel()))

print(np.mean(auc))

# if FLAGS.model_type == "sgd_svm":
#          model = SGDClassifier(max_iter=10000, tol=1e-10)
#      elif FLAGS.model_type == "svm":
#          model = LinearSVC(max_iter=10000, tol=1e-10)
