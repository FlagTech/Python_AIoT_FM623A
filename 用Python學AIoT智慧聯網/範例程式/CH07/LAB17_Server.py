from machine import I2C, Pin
import machine
import mpu6050
import time
from keras_lite import Model
import ulab as np
import network
import socket

host = '0.0.0.0'   # IP位址。0.0.0.0表示所有ip皆可以接收
port = 9999        # 通訊埠編號。 

mean = 2642.4535925925925          
std = 10607.848009804136             

model = Model('gesture_model.json')  
label_name = ['right','down','stop','left','up']

button=Pin(12,Pin.IN,Pin.PULL_UP)  
i2c = I2C(scl=Pin(25),sda=Pin(26))
accelerometer = mpu6050.accel(i2c)

data=[]

while(accelerometer.get_values()['AcX']==0 and
      accelerometer.get_values()['AcY']==0 and
      accelerometer.get_values()['AcZ']==0):  
    pass

sta=network.WLAN(network.STA_IF)
sta.active(True)   
sta.connect('無線網路名稱','無線網路密碼')   
while(not sta.isconnected()):
    pass
print('IP位址：',sta.ifconfig()[0])   

sock = socket.socket()
sock.bind((host, port))
sock.listen(1)   

print("等待客戶端連線")
(csock, adr) = sock.accept()
print(adr)
print("客戶端已連線")

while True:
    
    data=[]  # 重製data
    
    if(button.value()==0):  # 按下按鈕
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
        
        while(button.value()==0): # 等待放開按鈕
                pass  
        csock.send(direction)     # 發出[方向]訊號            
        time.sleep(0.1)           # 稍微暫停一下
sock.close()                      # 關閉socket