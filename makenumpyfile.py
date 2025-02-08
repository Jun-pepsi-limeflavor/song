from RawPreProcessing import rawpreprocessing
from Data_Extract import data_extraction
from SlidingWindow import slidingwindow
import numpy as np
# 클래스 인스턴스 생성
processor = rawpreprocessing(data_set_per_label=10, time_window=3) # label당 10개 , 창의 크기 3
num_data_set=processor.num_data_set
# 데이터 처리 및 시각화 반복
for i in range(1, num_data_set + 1):
    file_path = f"C:/Users/교육생-PC08/raw_data/RawData{i}.csv"
    
    # CSV 파일에서 양 끝 10% 데이터 제거
    result_df = processor.remove_edges_from_csv(file_path)

    # 데이터 시각화 & T초로 자르기
    processor.plot_csv_data(result_df)

    # 새로운 CSV 파일로 저장
    #trimmed_file_path = f"C:/Users/교육생-PC08/trimmed_data/trimmed_data{i}.csv"
    #result_df.to_csv(trimmed_file_path, index=False)

# 최종 3차원 배열 생성
total_array=processor.make_total_array()
y_label=processor.Y_label
np.save('full_data.npy', total_array)