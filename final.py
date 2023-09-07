import mysql.connector
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error





# example input
# (섭취 탄수화물)
carbohydrate_input = 88

# (사용자 id) 
member_id_input = 1


def perform_ai_calculations(carbohydrate, member_id):

    #aws mariaDB 연결 
    db_config = {
        'host':"user-info.cmvuhmicbso8.ap-southeast-2.rds.amazonaws.com",
        'user':"admin",
        'password':"Tkarnrtleo123!",
        'database':"user-info"
    }
    
    # 입력 array 초기화
    icr_update_values = []
    glucose_values = []
    goal_values=[]

    try: 
        # mariaDB로부터 사용자 id(member_id)에 맞게 목표 혈당, 혈당 기록, 인슐린-탄수화물비(ICR)를 가져온다
        connection_array = mysql.connector.connect(**db_config)
        cursor_array = connection_array.cursor()

        array_query = "SELECT goal, glucose, icr_update FROM result WHERE member_id = %s"
        cursor_array.execute(array_query, (member_id,))

        array_results = cursor_array.fetchall()

        for result in array_results:
            goal_values.append(result[0])
            glucose_values.append(result[1])
            icr_update_values.append(result[2])
        
        if array_results:
            # goal = 사용자의 목표 혈당
            # icr = 사용자의 인슐린-탄수화물비
            # glucose = 사용자의 혈당 기록 array
            goal_array = goal_values
            goal = goal_array[-1]

            icr_array = icr_update_values 
            icr = icr_array[-1]
            
            glucose = glucose_values
            
        else:
            print("사용자 데이터 베이스 error")

    finally:
        if connection_array:
            connection_array.close()
    
    # example 혈당 기록    
    # for stage 1
    # glucose = [150, 140, 160, 174]
    # for stage 2
    # glucose = [100, 110, 120, 130, 140, 180, 190, 180, 160, 170, 180, 176, 175, 174, 173]
    # for stage 3
    # glucose = [132, 131, 133, 131, 130, 130, 130, 131, 130, 130, 136, 140, 140, 143, 140, 132, 130, 131, 137, 144, 144, 146, 139, 147, 131, 133, 131, 130, 130, 130, 131, 130, 130, 136, 140, 140, 153, 152, 152, 150, 150, 150, 157, 146, 157, 157, 132, 135, 138, 130, 133, 157, 136, 136, 131, 135, 139, 155, 130, 132, 158, 152, 151, 137, 159, 155,131, 133, 131, 130, 130, 130, 131, 130, 130, 136, 140, 140, 154, 131, 158, 131, 135, 143, 143, 130, 141, 147, 141, 157, 136, 143, 150, 135, 135, 151, 132, 156, 135, 157, 160, 158, 144, 153, 160, 170, 175, 183, 182, 194, 196, 180, 189, 199, 184, 185, 191]



    # stage 1
    if len(glucose) < 5:
        update_icr = icr
    
    # stage 2
    elif len(glucose) >= 5 and len(glucose) < 100:
        df = pd.DataFrame({'Data_set': glucose})
        ema = df.ewm(com=0.5).mean()

        #결과 그래프 plot
        #plt.plot(glucose, label = "data")
        #plt.plot(ema, label = "EMA Values")
        #plt.xlabel("Days")
        #plt.ylabel("Blood Sugar Level")
        #plt.legend()
        #plt.show()
        #print(ema)


        #혈당 기록을 보고 다음 혈당 예측값 = predict
        predict = int(ema.iloc[-1])
        #print(predict)



        #다음 혈당이 목표 혈당에 비해 높게 예측 된다면 사용자의 인슐린-탄수화물비(ICR) 조정
        if predict > goal + 20:
            update_icr = icr - 1
        elif predict < goal - 20:
            update_icr = icr + 1
        else:
            update_icr = icr 



    # stage 3
    else:
        #array를 dataset matrix로 변환
        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)
        # numpy array로 변환
        dataset = np.array(glucose).reshape(-1, 1).astype('float32')
        
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        
        #train, test 셋 구분
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        
        look_back = 3
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        
        #인풋값 reshape [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        
        #LSTM 생성
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
        
        #예측 생성
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        
        #값 출력을 위해 다시 inverse transform
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        #다음 혈당 예측값 반올림하여 저장
        predict = round((testPredict[-1][0]))

        
        #MSE 계산
        trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        #print('Train Score: %.2f RMSE' % (trainScore))
        testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        #print('Test Score: %.2f RMSE' % (testScore))
        
        
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
        
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
        
        #데이터 예측 그래프 plot
        #plt.plot(scaler.inverse_transform(dataset))
        #plt.plot(trainPredictPlot)
        #plt.plot(testPredictPlot)
        #plt.show()
        
        #print(predict)


        #다음 혈당이 목표 혈당에 비해 높게 예측 된다면 사용자의 인슐린-탄수화물비(ICR) 조정
        if predict > goal + 10:
            update_icr = icr - 0.5
        elif predict < goal - 10:
            update_icr = icr + 0.5
        else:
            update_icr = icr
            
            

    #조정한 인슐린-탄수화물비(ICR)을 활용하여 적정 인슐린 투여량 계산, 반올림 후 저장
    insulin_result = carbohydrate/update_icr

    round_insulin_result = round(insulin_result)
    
    return round_insulin_result, carbohydrate, update_icr
    


    result = perform_ai_calculations(carbohydrate_input, member_id_input)
    print(result)