import socket
from pynput import keyboard
from pynput.keyboard import Key
import time

use_keyboard = keyboard.Controller()         # 建立鍵盤物件

sock=socket.socket()
sock.connect(('請輸入伺服端的 IP 位址', 9999)) # IP位址需要做更改

print('開始')

while True:
    data=sock.recv(1024)

    if(data==b"left"):
        use_keyboard.press(Key.left)
        use_keyboard.release(Key.left)
        print('left')

    elif(data==b"right"):
        use_keyboard.press(Key.right)
        use_keyboard.release(Key.right)
        print('right')

    elif(data==b"up"):
        use_keyboard.press(Key.up)
        use_keyboard.release(Key.up)
        print('up')

    elif(data==b"down"):
        use_keyboard.press(Key.down)
        use_keyboard.release(Key.down)
        print('down')
        
    elif(data==b"stop"):
        print('stop')
    
    data=''
    time.sleep(0.001)       # 稍微暫停一下
    
sock.close()                # 關閉 socket        
