import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

model_name = 'brits'

impute = np.load('./{}_data.npy'.format(model_name)).reshape(-1, 48 * 35)
label = np.load('./{}_label.npy'.format(model_name))#[3997:]

data = np.nan_to_num(impute)#[3997:]
print(data.shape)
n_train = 3000#data.shape[0] // 2

print(impute.shape)
print(label.shape)

# data = StandardScaler().fit_transform(data)

model = LogisticRegression().fit(data[:n_train], label[:n_train].ravel())
preds = model.predict_proba(data[n_train:])[:, 1]
auc = roc_auc_score(label[n_train:].ravel(), preds.ravel())
print(auc)