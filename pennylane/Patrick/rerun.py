from os import system
from time import sleep

counter = 0
while True:
    print("Starting run "+str(counter))
    system("python patrick.py")
    print("Restarting...")
    sleep(0.2)
    counter = +1