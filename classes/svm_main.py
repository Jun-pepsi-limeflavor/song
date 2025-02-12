from makenumpyfile import make_data_csv
from SVM_model import *

data_set, Y_label = make_data_csv(folder_path="C:/Users/user/OneDrive/바탕 화면/test/rawdataset", file_name="RawData", data_set_per_label=10, time_window=3)
test, _=make_data_csv(folder_path="C:/Users/user/OneDrive/바탕 화면/test/testdataset", file_name="TestData", data_set_per_label=5, time_window=3)

#슬라이딩 윈도우와 dataExtraction의 합작으로 데이터 추출 후 scale까지
X, y=train_feature_extract(data_set, Y_label)
label_encoder = LabelEncoder()
scaler = StandardScaler()
X_train, y_encoded=svm_train_scale(X, y, label_encoder, scaler)

#svm모델에 학습
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_encoded)

#test 데이터 받아오기
tests=test_feature_extract(test)
test_sample=svm_test_scale(tests, scaler)

#예측
y_pred = svm_model.predict(test_sample)
print(label_encoder.inverse_transform(y_pred))