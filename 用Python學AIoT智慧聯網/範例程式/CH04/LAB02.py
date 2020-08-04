from machine import Pin,ADC
import time

adc_pin=Pin(36)            # 36是 ESP32 的 VP 腳位
adc = ADC(adc_pin)         # 建立ADC物件
adc.width(ADC.WIDTH_9BIT)  # 設定ADC範圍。9BIT代表範圍是0~511
adc.atten(ADC.ATTN_11DB)   # 將最大感測電壓設定成3.6V, 超過3.6V時會得到ADC最大值511
while True:
    print(adc.read())      # 顯示ADC值
    time.sleep(0.5)        # 暫停0.5秒