import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

model_name = 'brits'

impute = np.load('./{}_data.npy'.format(model_name)).reshape(-1, 48 * 35)
label = np.load('./{}_label.npy'.format(model_name))[3997:]

data = np.nan_to_num(impute)[3997:]
print(data.shape)
n_train = data.shape[0] // 2

print(impute.shape)
print(label.shape)




# data = StandardScaler().fit_transform(data)

model = LogisticRegression(C=C, max_iter=10000, tol=1e-10).fit(data[:n_train], label[:n_train].ravel())
pred = model.decision_function(data[n_train:])
auc = roc_auc_score(label[n_train:].ravel(), pred.ravel())
