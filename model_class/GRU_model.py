import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from RawPreProcessing import rawpreprocessing
import Data_Extract
from SlidingWindow import slidingwindow


'''
변수 설정
batch_size = 몇 개의 묶음으로 학습을 시킬건지(유동)
input_size = 들어가는 input의 vector size (고정)
hidden_size = hidden layer의 size (반고정)
num_layers = (반고정)
output = 출력 값의 크기 (1로 고정)
learning_rate = 학습률 (유동)
num_epoch = 학습 횟수 (유동)
'''

# makenumpyfile이용

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



label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_tensor = torch.tensor(X, dtype=torch.float32)   # (batch_size, seq_length, input_dim)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# Dataset & DataLoader 설정
# dateset을 n개로 나눠서 최적화 진행
batch_size = 4
dataset = TensorDataset(X_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # 시계열 데이터니깐 shuffle을 하면 안되지만 sliding window를 사용했기때문에 여기선 True



class GRUMotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUMotionClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) # 분류 문제이기 때문에 outputsize를 받음. 만약 regression문제라면 1로 고정정

    def forward(self, x):
        gru_out, hidden = self.gru(x)  # hidden: (num_layers, batch_size, hidden_size)
        # 만약 gru_out이 2차원 텐서라면, 시퀀스 길이가 1인 경우일 수 있음
        if gru_out.ndimension() == 2:  # (batch_size, hidden_size)일 때
            out = self.fc(gru_out)  # 바로 fc 레이어에 전달
        else:  # (batch_size, seq_length, hidden_size)일 때
            out = self.fc(gru_out[:, -1, :])  # 마지막 시점의 hidden state 사용
            
        return out
    
model = GRUMotionClassifier(input_size=40, hidden_size=64, num_layers=2, output_size=1)
# input_size는 현재 x, y, z, a에서 뽑은 feature 40개
# hidden_size는 이전 데이터를 얼마나 기억할 것인지, 높으면 정확성이 올라가지만 너무 올라가면 과적합
# num_layers는 GRU 층
# output_size는 y_label의 개수(현재는 waving 밖에 없어서 1로함.)




# ========== 3. 학습 설정 ==========
learning_rate=0.001 # 학습률, weight를 update할때 얼만큼 weight를 조정할건지, 너무 크면 확확 바뀌고, 너무 작으면 찔끔찔끔 변화함. 적당한 것이 0.001
criterion = nn.CrossEntropyLoss()  # 분류 문제 -> CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 처음에는 큰 lr을 사용하다가 점차 작은 lr을 사용하는 최적화 알고리즘




# ========== 4. 학습 실행 ==========
num_epochs = 20

for epoch in range(num_epochs):
    for batch_X, batch_y in data_loader:
        optimizer.zero_grad() # 이전 Epoch에서 계산된 기울기(Gradient) 초기화
    
        # Forward
        outputs = model(batch_X) # 모델에 입력 데이터를 넣어 예측값 계산
        loss = criterion(outputs, batch_y)  # 손실(loss) 계산

        # Backward & Optimize
        loss.backward() # 역전파(Backpropagation) 수행하여 기울기 계산
        optimizer.step() # 가중치 업데이트

    if (epoch + 1) % 10 == 0: # 10번 주기로 학습이 잘되고 있는지 확인
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")




# ========== 5. 테스트 ==========
# test_sample = torch.randn(1, seq_length, input_dim)  # 임의의 테스트 데이터
model.eval()
with torch.no_grad():
    prediction = model(test_sample)
    predicted_class = torch.argmax(prediction, dim=1).item()
    print(f"Predicted Motion: {predicted_class}") # predicted 출력