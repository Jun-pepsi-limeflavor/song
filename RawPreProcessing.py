import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import math

class rawpreprocessing:
    def __init__(self, data_set=10):
        self.data_set = data_set
        self.raw_array = []  # 3차원 배열을 만들기 위한 리스트
        self.total_array = None  # 최종 3차원 배열
        self.Y_lable= None # 행동에 대한 라벨링 값 저장하는 리스트 
        self.set_data_set()

    def set_data_set(self):
        """사용자로부터 행동 개수를 입력받고, 해당 행동에 대한 라벨을 저장"""
        try:
            num_actions = 1 #(input("몇 가지 행동을 학습하시겠습니까? "))  # 행동 개수 입력 받기
            labels=[]
            for i in range(num_actions):
                action_label = "흔들기" #input(f"{i+1}번째 행동의 라벨을 입력하세요: ")  # 행동 라벨 입력 받기
                labels.append(action_label)
                
            # 리스트를 NumPy 배열로 변환하여 저장
            self.Y_lable = np.array(labels)

            # 데이터셋 크기를 행동 개수 * 10 으로 설정  
            self.data_set = num_actions * 10
            print(f"총 {self.data_set}개의 데이터를 처리합니다.")

        except ValueError:
            print("올바른 숫자를 입력해주세요.")
        
    def make_total_array(self):
        """
        raw_array를 3차원 NumPy 배열로 변환하는 함수.
        """
        if self.raw_array:
            self.total_array = np.stack(self.raw_array, axis=0)
            print("최종 3차원 배열 형태:", self.total_array.shape)
        else:
            print("읽은 데이터가 없습니다.")

    def remove_edges_from_csv(self, file_path):
        """
        CSV 파일에서 앞뒤 10% 데이터를 제거하는 함수.
        """
        df = pd.read_csv(file_path)
        total_rows = len(df)
        ten_percent = int(total_rows * 0.1)
        df_trimmed = df.iloc[ten_percent:-ten_percent]
        return df_trimmed

    def filter_data_by_time(self, df, x_value, time_window=3):
        """
        특정 X 값부터 time_window(기본 3초) 내의 데이터를 필터링하는 함수.
        """
        x_column = df.iloc[:, 0]
        start_index = (x_column >= x_value).idxmax()
        end_index = start_index 

        for i in range(start_index + 1, len(df)):
            if x_column[i] - x_value >= time_window:
                break
            end_index = i
        if start_index+300 > len(df):
            start_index=len(df)-300
            end_index=len(df)

        filtered_df = df.iloc[start_index:start_index + 300]
        return filtered_df

    def make_csv_array(self, file_name):
        """
        CSV 파일을 NumPy 배열로 변환하는 함수.
        """
        try:
            df = pd.read_csv(file_name)
            df.rename(columns={
                'Linear Acceleration x (m/s^2)': 'x',
                'Linear Acceleration y (m/s^2)': 'y',
                'Linear Acceleration z (m/s^2)': 'z',
                'Absolute acceleration (m/s^2)': 'a'
            }, inplace=True)
            df.drop(['Time (s)'], axis=1, inplace=True)

            numppy = df.to_numpy()
            print(f"파일 '{file_name}' 처리 완료!")
            return numppy
        except FileNotFoundError:
            print(f"파일 '{file_name}'이(가) 존재하지 않습니다.")
            return None

    def plot_csv_data(self, file_path, i):
        """
        CSV 파일 데이터를 그래프로 시각화하는 함수.
        """
        df = pd.read_csv(file_path)
        x_values = df.iloc[:, 0]
        y_values = df.iloc[:, 1:]

        plt.figure(figsize=(10, 6))
        lines = []

        for column in y_values.columns:
            line, = plt.plot(x_values, y_values[column], label=column)
            lines.append(line)

        plt.title(f'{i}th CSV Data Visualization')
        plt.xlabel(df.columns[0])
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)

        mplcursors.cursor(lines, hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(f'시간: {sel.target[0]:.2f}')
        )

        def on_click(event):
            if event.inaxes:
                clicked_x = event.xdata
                print(f"Clicked X Value: {clicked_x:.2f}")

                # 기존 filtered_df를 processed_array로 변환
                filtered_df = self.filter_data_by_time(df, clicked_x, time_window=3)
                print("Filtered Data:")
                print(filtered_df)

            # NumPy 배열로 변환
                processed_array = filtered_df.to_numpy()
                processed_array=processed_array[:,1:]
                print("Processed Array:")
                print(processed_array)

            # 새로운 데이터를 raw_array에 추가
                self.raw_array.append(processed_array)
            
            # 최종 3차원 배열 갱신
                #self.make_total_array()
                plt.close()

        plt.gcf().canvas.mpl_connect("button_press_event", on_click)
        plt.show()

    

    def fourier_trans(self, data_signal, sampling_rate): #fft를 통해 한 축의 가속도 그래프에서 freq와 amp를 반환
        amp=np.fft.fft(data_signal)
        freq=np.fft.fftfreq(len(data_signal), d=1/sampling_rate)
    
        low_frq_limit = 10
        #인간의 한계
    
        v_freq = (freq >= 0) & (freq <= low_frq_limit)
        a_amp = np.abs(amp)

        valid_amp = a_amp[v_freq]
        valid_freq = freq[v_freq]
        print("valid_amp")
        print(valid_amp)
        max_index=1
        for i in range (1,len(valid_amp)):
            if(valid_amp[max_index]<valid_amp[i]):
                max_index=i
        max_freq = valid_freq[max_index]  # 해당 인덱스의 valid_freq 값 반환
        return max_freq
     
    

    def sliding_window(self, T=1, n=0.4, i=0):
        """ 슬라이딩 윈도우 방식으로 데이터를 잘라 반환하는 함수
        T: 한 윈도우의 크기 (초 단위, 1초 = 100개의 열)
        n: 슬라이딩 간격 (초 단위)
        i: self.total_array에서 몇 번째 데이터를 사용할지 지정
        """
        data = self.total_array[i]  # i번째 Raw Data 선택
        num_columns = data.shape[0]  # 행 개수 가져오기
        window_size = T * 100  # 한 번에 자를 크기
        step_size = n * 100  # 슬라이딩 간격

    # 실행 횟수 계산
        max_steps = math.ceil((num_columns - window_size)/step_size + 1) 


    # 슬라이딩 윈도우 실행
        for step in range(max_steps):
            
            start_roW= step * step_size
            start_roW=int (start_roW)

            end_roW = start_roW + window_size
            end_roW=int (end_roW)
            if(num_columns-start_roW<T*100):
                break
            else:
                Raw_data_cut = data[start_roW:end_roW,:]  # 특정 열 범위 선택
                print(f"Step {step+1}: Raw_data_cut shape = {Raw_data_cut.shape}")
                print(Raw_data_cut)
            # 마지막 데이터 부족할 경우 처리
        if num_columns - (max_steps * step_size) < window_size:
                window_size=int(window_size)
                Raw_data_cut = data[num_columns-window_size-1:num_columns-1, :]  # 마지막 window_size만큼 가져오기
                print(f"Final Step: Raw_data_cut shape = {Raw_data_cut.shape}")
                print(Raw_data_cut)


    
       
   
    



        