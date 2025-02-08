import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from RawPreProcessing import rawpreprocessing
import Data_Extract
from SlidingWindow import slidingwindow

#앞의 10번 연타하는거는 알아서 하시고~우
#input으로 label받는 것도 구현하고고
data_set=np.load('full_data.npy')
Y_label=["Waving"]
num=10 #data_set의 개수

X=[]
y=[]

sliding_window_processor = slidingwindow(data_set, Y_label)
for j in range(0, num):  # row data 갯수 만큼 돌림
        part_data = data_set[j]

        # Fourier 변환을 통해 최대 주파수 구하기
        max_freq = sliding_window_processor.fourier_trans_max_amp(part_data[:, 3], 100)  # absolute 값
        #print(f"Max Frequency for dataset {j}: {max_freq}")

        # SlidingWindow 클래스 인스턴스 생성 및 슬라이딩 윈도우 처리
        win_datas=sliding_window_processor.sliding_window(1/max_freq,1,j)
        #print(Data_Extract.data_extraction(win_datas[3]).extract_feature())
        for i in range(0, len(win_datas)):
                
                X.append(Data_Extract.data_extraction(win_datas[i]).extract_feature())
                y.append(Y_label[int(j/10)])

#data랑 label붙는 곳
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# 모델 평가
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'테스트 정확도: {accuracy:.4f}')

#csv파일로 꺼내서 보는게 편하면 그렇게 바꿔도 됨.
print(y_test, y_pred)