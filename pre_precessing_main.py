from RawPreProcessing import rawpreprocessing
from Data_Extract import data_extraction
from SlidingWindow import slidingwindow

# 클래스 인스턴스 생성
processor = rawpreprocessing(data_set=10)

# 데이터 처리 및 시각화 반복
for i in range(1, processor.data_set + 1):
    file_path = f"/Users/songjunha/Downloads/Acceleration_without_g_2025-01-28_12-20-37/RawData{i}.csv"
    
    # CSV 파일에서 양 끝 10% 데이터 제거
    result_df = processor.remove_edges_from_csv(file_path)
    
    # 새로운 CSV 파일로 저장
    trimmed_file_path = f"/Users/songjunha/Downloads/Acceleration_without_g_2025-01-28_12-20-37/remove_edge_Data{i}.csv"
    result_df.to_csv(trimmed_file_path, index=False)
    
    print(result_df)

    # 데이터 시각화
    processor.plot_csv_data(trimmed_file_path, i)
    
    # NumPy 배열 변환 후 리스트에 추가
    processed_array = processor.make_csv_array(trimmed_file_path)

# 최종 3차원 배열 생성
processor.make_total_array()
print("make_total_array 의 shape:")
print(processor.total_array.shape)


#sliding window & FFT 

sliding_window_processor = slidingwindow(processor.total_array,processor.Y_lable)

for j in range(0, processor.data_set):  # row data 갯수 만큼 돌림
        part_data = processor.total_array[j]

        

        # Fourier 변환을 통해 최대 주파수 구하기
        max_freq = sliding_window_processor.fourier_trans_max_amp(part_data[:, 3], 100)  # absolute 값
        print(f"Max Frequency for dataset {j}: {max_freq}")

        # SlidingWindow 클래스 인스턴스 생성 및 슬라이딩 윈도우 처리
        sliding_window_processor.sliding_window(1/max_freq,1,j)