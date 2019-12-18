import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
data = pd.read_csv("result.csv")
df = data.drop(["bikeNo", "createDate", "typefixResult", "userGuid", "userHasMonthCard", "maintUserGuid", "maintUserName", "fixResult", "updateDate"], axis=1)
y = df.pop("bikefixResult")
X = df.as_matrix()
y = y.as_matrix()


# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 设置参数
clf = xgb.XGBClassifier(learning_rate = 0.14,
                        n_estimators = 1000,
                        max_depth = 5,
                        gamma = 0,
                        subsample = 0.8,
                        colsample_bytree = 0.8,
                        nthread = 4)
# 训练
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy : %.2f" % metrics.accuracy_score(y_test, y_pred))
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, test_2)) ##必须二分类才能计算
