import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

model_name = 'brits'

impute = np.load('./{}_data.npy'.format(model_name)).reshape(-1, 48 * 35)
label = np.load('./{}_label.npy'.format(model_name))

data = np.nan_to_num(impute)

n_train = 2000

print(impute.shape)
print(label.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

auc = []

#data = StandardScaler().fit_transform(data)

for i in range(20):
    model =  RandomForestClassifier().fit(data[:n_train], label[:n_train])
    preds = model.predict_proba(data[n_train:])[:, 1]
    auc.append(roc_auc_score(label[n_train:].ravel(), preds.ravel()))

print(np.mean(auc))
