from machine import Pin,ADC
import time
from keras_lite import Model  
import ulab as np
import network     # 匯入network模組
import urequests   # 匯入urequests模組
from umqtt.robust import MQTTClient

# 連線至無線網路
sta=network.WLAN(network.STA_IF)
sta.active(True)   
sta.connect('無線網路名稱','無線網路密碼')  # 更換為無線網路帳號、密碼 

while not sta.isconnected() :
    pass

print('Wifi連線成功')

# Adafruit IO 設定
aio_username = "你的Adafruit用戶名"
aio_key = "你的AIO密鑰"

mqtt_client = MQTTClient(
    client_id=aio_username,
    server="io.adafruit.com",
    user=aio_username,
    password=aio_key,
    ssl=False)
url_line="MAKE 腳本網址"

mean = 170.98275862068965
std = 90.31162360353873
model = Model('temperature_model.json')

adc_pin = Pin(36)
adc = ADC(adc_pin)
adc.width(ADC.WIDTH_9BIT)
adc.atten(ADC.ATTN_11DB)

mqtt_client.connect()
print("MQTT 連線成功")

while True:
    data = 0
    for i in range(20):
        thermal = adc.read()
        data += thermal
        time.sleep(0.01)

    data = data / 20
    data = np.array([int(data)])
    data = data - mean
    data = data / std
    tem = model.predict(data)
    tem = round(tem[0] * 100, 1)
    print(tem, end='   ')

    # 將數據發布到 Adafruit IO
    topic = aio_username + "/feeds/temperature"
    mqtt_client.publish(topic, str(tem))
    print("數據已發布至 Adafruit IO")
    
    if(tem>=37.5):   # 當體溫超過37.5度時, 傳LINE做警告
        r = urequests.get(url_line+"?value1="+str(tem)) # 傳送至LINE
        r.close()
        print("警告!!!發燒了!!!")

    time.sleep(60)  # 暫停60秒    
