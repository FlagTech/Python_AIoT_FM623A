from machine import I2C, Pin
import mpu6050
import time
from keras_lite import Model
import ulab as np

mean= 2642.4535925925925   # 平均數
std= 10607.848009804136    # 標準差

model = Model('gesture_model.json')  #呼叫權重檔
label_name = ['right','down','stop','left','up']

button=Pin(12,Pin.IN,Pin.PULL_UP)  # 開始按鈕
i2c = I2C(scl=Pin(25),sda=Pin(26))
accelerometer = mpu6050.accel(i2c)

data=[]     

while(accelerometer.get_values()['AcX']==0 and
      accelerometer.get_values()['AcY']==0 and
      accelerometer.get_values()['AcZ']==0):  
    pass

print('開始')

while True:
    data=[]  # 重製data
    
    if (button.value()==0):  # 按下按鈕
        for i in range(10):
            six_data = accelerometer.get_values()
            
            data.append(six_data['AcX'])    
            data.append(six_data['AcY'])    
            data.append(six_data['AcZ'])    
            data.append(six_data['GyX'])    
            data.append(six_data['GyY'])    
            data.append(six_data['GyZ'])    
            
        data=np.array([data])
        data=data-mean
        data=data/std
        output = model.predict_classes(data)
        direction = label_name[output[0]]
        print(direction)
        
        while(button.value()==0): 
                pass
            
        time.sleep(0.1) # 稍微暫停一下