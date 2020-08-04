from pynput.keyboard import Controller
from pynput.keyboard import Key
import time

my_keyboard=Controller()

my_keyboard.press(Key.up)    # 按下↑鍵
my_keyboard.release(Key.up)  # 放開↑鍵
time.sleep(1)

my_keyboard.press(Key.down)    # 按下↓鍵
my_keyboard.release(Key.down)  # 放開↓鍵
time.sleep(1)

my_keyboard.press(Key.left)    # 按下←鍵
my_keyboard.release(Key.left)  # 放開←鍵
time.sleep(1)

my_keyboard.press(Key.right)    # 按下→鍵
my_keyboard.release(Key.right)  # 放開→鍵
time.sleep(1)