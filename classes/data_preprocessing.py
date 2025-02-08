
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors 
import pandas as pd
import numpy as np



allArray = [] # data nparry 3차원 배열로 예상됨
data_set=10 #입력 받는 데이터 갯수 10개로 한정. 

def remove_edges_from_csv(file_path ):
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
    # 전체 행 수 구하기
    total_rows = len(df)
    
    # 10%에 해당하는 행 수 계산
    ten_percent = int(total_rows * 0.1)
    
    # 앞 10%와 뒤 10% 데이터를 제외한 나머지 데이터
    df_trimmed = df.iloc[ten_percent:-ten_percent]
    
    # 결과 출력 또는 파일로 저장 (필요에 따라 선택)
    return df_trimmed

# 사용 예시


def display_text(message, font_size=20, box_color='lightblue'):
     fig, ax = plt.subplots(figsize=(10, 6))
     ax.axis('off')  # 축 숨기기
    
    # 텍스트 표시
     ax.text(
        0.5, 0.5, message,  # 텍스트 내용 및 위치 (중앙)
        fontsize=font_size,  # 텍스트 크기
        color='black',       # 텍스트 색상
        #bbox=dict(facecolor=box_color, alpha=0.5),  # 텍스트 박스 색상과 투명도
        ha='center', va='center'  # 텍스트 정렬 (가로, 세로)
    )
    
    # 그래프 표시
     plt.show()
     


def filter_data_by_time(df, x_value, time_window=3):
    """
    특정 X값부터 3초 내의 데이터를 필터링합니다.
    time_window는 기본값 3초로 설정됨.
    """
    # 첫 번째 열을 X값으로 사용 (시간값)
    x_column = df.iloc[:, 0]  # 첫 번째 열이 X 값 (시간 값)
    
    # X 값 기준으로 필터링: X 값보다 크거나 같은 첫 번째 인덱스 찾기
    start_index = (x_column >= x_value).idxmax()  # 클릭한 X 값보다 크거나 같은 첫 번째 인덱스
    
    # 3초 이내에 해당하는 마지막 인덱스를 찾기
    end_index = start_index
    for i in range(start_index + 1, len(df)):
        if x_column[i] - x_value >= time_window:
            break
        end_index = i  # 3초 미만이라면 계속해서 마지막 데이터까지 포함
    
    # 추출된 데이터 리턴
    filtered_df = df.iloc[start_index:end_index + 1] 
    
   # display_text(
    #message="Data Selected",  # 텍스트 내용
    #font_size=30,            # 텍스트 크기
    #box_color='yellow'       # 텍스트 박스 색상
#)
    plt.close()
    filtered_df.to_csv('/Users/songjunha/Desktop/Code_Name/2025_New_shit/contents/testing_for_microbit/F_trimmed_file1.csv ', index=None)
    
    
    return filtered_df
def make_csv_array(file_name):
    try:
        # CSV 파일을 읽고 NumPy 배열로 변환
        df = pd.read_csv(file_name)
        df.rename(columns={'Linear Acceleration x (m/s^2)':'x'}, inplace=True)
        df.rename(columns={'Linear Acceleration y (m/s^2)':'y'}, inplace=True)
        df.rename(columns={'Linear Acceleration z (m/s^2)':'z'}, inplace=True)
        df.rename(columns={'Absolute acceleration (m/s^2)':'a'}, inplace=True)
        df.drop(['Time (s)'], axis=1, inplace=True)

        numppy = df.to_numpy()  # DataFrame을 NumPy 배열로 변환
        print(f"파일 '{file_name}' 처리 완료!")
    except FileNotFoundError:
        print(f"파일 '{file_name}'이(가) 존재하지 않습니다.")
    
    return numppy

def plot_csv_data(file_path, i):
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
    # 첫 번째 열을 X축 값으로 설정 (첫 번째 열은 X 값이므로 'df.iloc[:, 0]'을 사용)
    x_values = df.iloc[:, 0]
    
    # 첫 번째 행을 제외한 나머지 열들은 Y축 값으로 사용
    y_values = df.iloc[:, 1:]
    
    # 그래프 그리기
    plt.figure(figsize=(10, 6))  # 그래프 사이즈 설정
    
    lines = []
    
    # 각 열에 대해 그래프를 그림
    for column in y_values.columns:
        line, = plt.plot(x_values, y_values[column], label=column)  # line 객체를 반환받음
        lines.append(line)
    
    # 그래프 제목과 레이블 설정
    plt.title(f'{i}th CSV Data Visualization ')  
    plt.xlabel(df.columns[0])  
    plt.ylabel('Values')  
    plt.legend()  
    plt.grid(True)  
    
    # 마우스 커서 옆에 X 값을 표시하기 위한 객체
    cursor =  mplcursors.cursor(lines, hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f'X: {sel.target[0]:.2f}')
    )


    # 커서 옆에 X 값 표시
    #def on_hover(event):
    #    if event.artist is not None:
    #       # event.target이 numpy.ndarray 이므로 해당 값에 직접 접근
    #       x_value = event.target[event.target.index]  # 클릭된 X 값
    #       event.annotation.set_text(f'X: {x_value:.2f}')  # X 값 표시
    
    #cursor.connect("add", on_hover)
    
    # 클릭 이벤트 처리
    #def on_click(event):
       # if event.artist is not None:
            # event.target이 numpy.ndarray 이므로 해당 값에 직접 접근
        #    x_value = event.target[event.target.index]  # 클릭된 X 값
         #   print(f"클릭한 X값: {x_value}")
        
            # X 값 기준으로 3초 내의 데이터를 필터링
         #   filtered_df = filter_data_by_time(df, x_value)
        
            # 결과를 새로운 CSV 파일로 저장
        #  output_filename = f"filtered_data_{x_value}.csv"
        #   filtered_df.to_csv(output_filename, index=False)
        #   print(f"{output_filename} 파일이 생성되었습니다.")
    
    # 클릭 이벤트 연결
    #cursor.connect("add", on_click)
    # 그래프 표시
      
    def on_click(event):
        if event.inaxes:  # 클릭이 그래프 안에서 발생했는지 확인
            clicked_x = event.xdata  # 클릭한 X 좌표
            print(f"Clicked X Value: {clicked_x:.2f}")
        # filter_data_by_time 호출
            filtered_df = filter_data_by_time(df, clicked_x, time_window=3)
            print("Filtered Data:")
            print(filtered_df)

    
    # 클릭 이벤트 연결
    plt.gcf().canvas.mpl_connect("button_press_event", on_click)
    plt.show() 
    
# 사용 예시
for i in range(1,11): # 10번 반복 전처리 과정 반복. 
    file_path = f"/Users/songjunha/Downloads/Acceleration_without_g_2025-01-28_12-20-37/RawData{i}.csv" # i번째 CSV파일 읽어옴
    result_df = remove_edges_from_csv(file_path)

# 양 끝값 제거 결과를 새로운 CSV 파일로 저장 
    result_df.to_csv(f"/Users/songjunha/Downloads/Acceleration_without_g_2025-01-28_12-20-37/remove_edge_Data{i}.csv", index=False)
# 결과 확인 
    print(result_df)
    file_path = f"/Users/songjunha/Downloads/Acceleration_without_g_2025-01-28_12-20-37/remove_edge_Data{i}.csv"  # 원하는 파일 경로로 수정하세요
    plot_csv_data(file_path,i)
    allArray.append(make_csv_array(file_path))

if allArray:
    final_3d_array = np.stack(allArray, axis=0)  # axis=0: 첫 번째 축 기준으로 쌓기
    print("최종 3차원 배열 형태:", final_3d_array.shape)  # 배열 형태 출력
else:
    print("읽은 데이터가 없습니다.")

