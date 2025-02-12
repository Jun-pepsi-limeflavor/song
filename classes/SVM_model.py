import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import Data_Extract
from SlidingWindow import slidingwindow

def train_feature_extract(data_set, Y_label):
    X=[]
    y=[]
    num=len(data_set)
    sliding_window_processor = slidingwindow(data_set, Y_label)

    for j in range(0, num):  # row data 갯수 만큼 돌림
        part_data = data_set[j]

        # Fourier 변환을 통해 최대 주파수 구하기
        max_freq = sliding_window_processor.fourier_trans_max_amp(part_data[:, 3], 100)  # absolute 값

        # SlidingWindow 클래스 인스턴스 생성 및 슬라이딩 윈도우 처리
        win_datas=sliding_window_processor.sliding_window(1/max_freq,1/max_freq*0.5,j)

        for i in range(0, len(win_datas)):
                
             X.append(Data_Extract.data_extraction(win_datas[i]).extract_feature())
             y.append(Y_label[int(j/10)])
    return X, y

def svm_train_scale(X, y, label_encoder, scaler):
    
    y_encoded = label_encoder.fit_transform(y)
    X_train = scaler.fit_transform(X)
    return X_train, y_encoded


def test_feature_extract(test):
    tests=[]
    sliding_window_test = slidingwindow(test, Y_label)
    for j in range(0, len(test)):  # row data 갯수 만큼 돌림
        part_data = test[j]

        # Fourier 변환을 통해 최대 주파수 구하기
        max_freq = sliding_window_test.fourier_trans_max_amp(part_data[:, 3], 100)  # absolute 값
        #print(f"Max Frequency for dataset {j}: {max_freq}")

        # SlidingWindow 클래스 인스턴스 생성 및 슬라이딩 윈도우 처리
        win_datas=sliding_window_test.sliding_window(1/max_freq,1/max_freq*0.5,j)
        tests.append(Data_Extract.data_extraction(win_datas[len(win_datas)//2]).extract_feature())
    return tests

def svm_test_scale(tests, scaler):
    test_sample=scaler.transform(tests)
    return test_sample



data_set=np.load('train_data.npy')
num=len(data_set)
Y_label = ['handshaking', 'punching', 'waving', 'walking', 'running']

X, y=train_feature_extract(data_set, Y_label)
label_encoder = LabelEncoder()
scaler = StandardScaler()

X_train, y_encoded=svm_train_scale(X, y, label_encoder, scaler)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_encoded)

test=np.load('test_data.npy')
tests=test_feature_extract(test)
test_sample=svm_test_scale(tests, scaler)

y_pred = svm_model.predict(test_sample)
#accuracy = accuracy_score(y_test, y_pred)
#print(f'테스트 정확도: {accuracy:.4f}')
print(label_encoder.inverse_transform(y_pred))