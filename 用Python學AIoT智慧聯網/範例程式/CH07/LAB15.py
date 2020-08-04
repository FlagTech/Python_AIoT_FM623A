from machine import I2C, Pin
import mpu6050
import time

button=Pin(12,Pin.IN,Pin.PULL_UP)  # 開始按鈕
i2c = I2C(scl=Pin(25),sda=Pin(26))
accelerometer = mpu6050.accel(i2c)

f=open('up.txt','w')    # 開啟txt檔
data=[]     

while(accelerometer.get_values()['AcX']==0 and
      accelerometer.get_values()['AcY']==0 and
      accelerometer.get_values()['AcZ']==0):  
    pass

# 儲存100筆
for j in range(100):    
    print('第'+str(j+1)+'筆')
    data=[]   
   
    while True:       
        if (button.value()==0):  # 按下按鈕
            for i in range(10):  # 連續10次
                six_data = accelerometer.get_values()
                
                data.append(six_data['AcX'])    # 加速度計 x 軸
                data.append(six_data['AcY'])    # 加速度計 y 軸
                data.append(six_data['AcZ'])    # 加速度計 z 軸
                data.append(six_data['GyX'])    # 陀螺儀 x 軸
                data.append(six_data['GyY'])    # 陀螺儀 y 軸
                data.append(six_data['GyZ'])    # 陀螺儀 z 軸
                
            f.write(str(data)[1:-1])  # data存到檔案中
            f.write("\n")             # 換行字元
            print('Save')
            print('')
            while(button.value()==0): # 放開按鈕
                pass
            time.sleep(0.1)
            break            
            
f.close()  # 關閉txt檔
