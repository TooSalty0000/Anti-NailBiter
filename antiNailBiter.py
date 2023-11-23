import tkinter as tk
from threading import Thread
import time
import cv2
import numpy as np
from keras.models import load_model

model = load_model(
    "/Users/12salty/Documents/Coding/Python Projects/Anti-NailBiter/model/keras_model.h5",
    compile=False,
)

# Load the labels
class_names = open(
    "/Users/12salty/Documents/Coding/Python Projects/Anti-NailBiter/model/labels.txt",
    "r",
).readlines()

camera = cv2.VideoCapture(0)


def bite_detection():
    global camera
    # Grab the webcamera's image.
    ret, image = camera.read()
    height = len(image)
    image = image[0:1080, height // 2 : height + height // 2]
    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


class App:
    def __init__(self, root):
        self.root = root
        root.geometry("400x250")  # Set the size of the window

        self.running = False

        self.start_button = tk.Button(
            root, text="Start", command=self.toggle_main_function
        )
        self.start_button.pack(pady=20)  # Add some padding for better layout
        self.start_button.config(width=20, height=5)  # Set the size of the button

        self.quit_button = tk.Button(root, text="Quit", command=root.destroy)
        self.quit_button.pack()
        self.quit_button.config(width=20, height=5)  # Set the size of the button

    def main_function(self):
        while self.running:
            print("Main function is running...")
            time.sleep(1)  # Simulating work

    def toggle_main_function(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self.main_function)
            self.thread.start()
            self.start_button.config(text="Stop")
        else:
            self.running = False
            self.thread.join()
            self.start_button.config(text="Start")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
