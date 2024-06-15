import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# 读取数据
train = pd.read_csv("C:/Users/23206/PycharmProjects/petfinder/petfinder-adoption-prediction/train/train.csv")
test = pd.read_csv("C:/Users/23206/PycharmProjects/petfinder/petfinder-adoption-prediction/test/test.csv")

# 填充缺失值
imp = SimpleImputer(missing_values=np.nan, strategy='median')
train["AdoptionSpeed"] = imp.fit_transform(train[["AdoptionSpeed"]])
train["AdoptionSpeed"] = train["AdoptionSpeed"].astype(int)

# 选取特征
x = train[['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
           'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
           'Quantity', 'Fee', 'State', 'VideoAmt', 'PhotoAmt']]
y = train['AdoptionSpeed']

# 分割数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# 决策树模型
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dt_predict = dtc.predict(x_test)
print(dtc.score(x_test, y_test))
print(classification_report(y_test, dt_predict, target_names=["0", "1", "2", "3", "4"]))

# 随机森林模型
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train)
rfc_y_predict = rfc.predict(x_test)
print(rfc.score(x_test, y_test))

# 使用全量数据进行训练和预测
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x, y)
x_test = test[['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
               'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
               'Quantity', 'Fee', 'State', 'VideoAmt', 'PhotoAmt']]
final_result = rfc.predict(x_test)
submission_df = pd.DataFrame(data={'PetID': test['PetID'].tolist(), 'AdoptionSpeed': final_result})
submission_df.to_csv('submission.csv', index=False)
