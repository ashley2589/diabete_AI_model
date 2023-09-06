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


#### 디비에서 받는것 ####
# amount = 인슐린 투여량 기록
# glucose = 혈당 기록
# carbohydrate_sum = 탄수 섭취 기록
# icr = icr
# goal = 목표 혈당


#### 백에서 받는것 ####
# carbohydrate = 매 끼 받는 탄수
# member_id = 멤버 아이디


#### 계산 결과 값 ####
# ai_result = 인슐린 투여량 계산값
#####################





#########################################################################################################################
### JAVA랑 연동하는부분 ###
app = Flask(__name__)


@app.route('/', methods=['POST'])
def calculate_result():
    data = request.get_json()
    carbohydrate = data['carb']
    member_id = data['memberId']

   
    ai_result = perform_ai_calculations(carbohydrate, member_id)
    print(ai_result)

    return jsonify(result=ai_result)
    ## ai_result = (인슐린 투여량, 매 끼 탄수합, update_icr)

#########################################################################################################################


#carbohydrate = 88
#member_id = 1


#########################################################################################################################
#################### 인슐린 투여량 계산식, 줄 250까지 
def perform_ai_calculations(carbohydrate, member_id, snack):
    db_config = {
        'host':"user-info.cmvuhmicbso8.ap-southeast-2.rds.amazonaws.com",
        'user':"admin",
        'password':"Tkarnrtleo123!",
        'database':"user-info"
    }
    
    #### array 초기화
    icr_update_values = []
    glucose_values = []
    goal_values=[]

    try: 
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
            goal_array = goal_values
            goal = goal_array[-1]

            icr_array = icr_update_values 
            icr = icr_array[-1]
            
            glucose = glucose_values
            
            
        else:
            print("기록지 error")

    finally:
        if connection_array:
            connection_array.close()
    ##############################################################
    ## 이제부터
    ## icr, goal 값 하나 int
    ## glucose = 혈당 기록 array 형태 [150, 160, ...]
    ##
    ## carbohydrate = 매 끼 탄수 int
    ##############################################################
    #### 탄수합이랑 주사 계산량 디비에 다시 넣기

        
    #glucose = [150, 140, 160, 174]
    #glucose = [100, 110, 120, 130, 140, 180, 190, 180, 160, 170, 180, 176, 175, 174, 173]
    #glucose = [132, 131, 133, 131, 130, 130, 130, 131, 130, 130, 136, 140, 140, 143, 140, 132, 130, 131, 137, 144, 144, 146, 139, 147, 131, 133, 131, 130, 130, 130, 131, 130, 130, 136, 140, 140, 153, 152, 152, 150, 150, 150, 157, 146, 157, 157, 132, 135, 138, 130, 133, 157, 136, 136, 131, 135, 139, 155, 130, 132, 158, 152, 151, 137, 159, 155,131, 133, 131, 130, 130, 130, 131, 130, 130, 136, 140, 140, 154, 131, 158, 131, 135, 143, 143, 130, 141, 147, 141, 157, 136, 143, 150, 135, 135, 151, 132, 156, 135, 157, 160, 158, 144, 153, 160, 170, 175, 183, 182, 194, 196, 180, 189, 199, 184, 185, 191]

    if len(glucose) < 5:
        update_icr = icr
        
    elif len(glucose) >= 5 and len(glucose) < 100:
        df = pd.DataFrame({'Data_set': glucose})
        ema = df.ewm(com=0.5).mean()
        #plt.plot(glucose, label = "data")
        #plt.plot(ema, label = "EMA Values")
        #plt.xlabel("Days")
        #plt.ylabel("Blood Sugar Level")
        #plt.legend()
        #plt.show()
        #print(ema)

        predict = int(ema.iloc[-1])



        print(predict)
        if predict > goal + 20:
            update_icr = icr - 1
        elif predict < goal - 20:
            update_icr = icr + 1
        else:
            update_icr = icr 




    else:
        # convert an array of values into a dataset matrix
        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)
        # Convert data to a NumPy array and reshape to 2D
        dataset = np.array(glucose).reshape(-1, 1).astype('float32')
        
        # Normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        
        # Split into train and test sets
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        
        # reshape into X=t and Y=t+1
        look_back = 3
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        
        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
        
        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # Print the most recent prediction
        predict = round((testPredict[-1][0]))

        
        # calculate root mean squared error
        trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        #print('Train Score: %.2f RMSE' % (trainScore))
        testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        #print('Test Score: %.2f RMSE' % (testScore))
        
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
        
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
        
        # plot baseline and predictions
        #plt.plot(scaler.inverse_transform(dataset))
        #plt.plot(trainPredictPlot)
        #plt.plot(testPredictPlot)
        #plt.show()


        print(predict)

        if predict > goal + 10:
            update_icr = icr - 0.5
        elif predict < goal - 10:
            update_icr = icr + 0.5
        else:
            update_icr = icr
            
            

            
    insulin_result = carbohydrate/update_icr
    round_insulin_result = round(insulin_result)
    
    return round_insulin_result, carbohydrate, update_icr

    #print(round_insulin_result)
#########################################################################################################################

if __name__ == '__main__':
    app.run()
