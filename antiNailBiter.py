import tkinter as tk
from threading import Thread, Lock
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
from keras.models import load_model
import NailBiteDetector as nbd


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the labels
class_names = open(
    "/Users/12salty/Documents/Coding/Python Projects/Anti-NailBiter/model/labels.txt",
    "r",
).readlines()

camera = cv2.VideoCapture(0)
camera_lock = Lock()


def bite_detection(app):
    global camera, class_names
    with camera_lock:
        if not camera.isOpened():  # Check if camera is available
            return
        ret, image = camera.read()
    if ret or app.running:
        # resize the image to 224x224
        height = len(image)
        width = len(image[0])
        # print("height:", height)
        image = image[0:height, (width - height) // 2 : height + (width - height) // 2]
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = cv2.flip(image, 1)
        cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predicts the model
        class_name, confidence = nbd.analyzeImage(cv2image, class_names)

        # Print the results
        # print("class_name:", class_name, "confidence:", confidence)
        # print(class_names)
        # if class_name == class_names[1].strip("\n"):
        #     print("Bite detected with confidence of", confidence)

        img = Image.fromarray(cv2image)
    else:
        img = Image.new("RGB", (224, 224), (0, 0, 0))  # Black box

    imgtk = ImageTk.PhotoImage(image=img)
    app.display_image(imgtk)
    return class_name == class_names[1]


def stop_thread(thread):
    if thread is not None:
        thread.join()


class App:
    def __init__(self, root):
        self.root = root
        root.geometry("400x600")  # Adjust the size of the window
        self.root.configure(background="systemWindowBackgroundColor")
        self.running = False

        # Initialize the title label
        self.title_label = tk.Label(root, text="Anti-NailBiter", font=("Arial", 30))
        self.title_label.pack(pady=20)

        # Initialize the image label with a black box
        initial_img = Image.new("RGB", (224, 224), (0, 0, 0))
        initial_imgtk = ImageTk.PhotoImage(image=initial_img)
        self.image_label = tk.Label(root, image=initial_imgtk)
        self.image_label.pack()
        self.image_label.imgtk = initial_imgtk
        self.image_label.configure(image=initial_imgtk)

        # Container frame for start button and sensitivity slider
        control_frame = tk.Frame(root)
        control_frame.pack(pady=20)

        # Initialize the start button
        self.start_button = tk.Button(
            control_frame, text="Start", command=self.toggle_main_function
        )
        self.start_button.grid(row=0, column=0, rowspan=2, padx=(0, 10))  # Left side
        self.start_button.config(width=10, height=3)

        # Initialize the sensitivity slider
        self.sensitivity_label = tk.Label(
            control_frame, text="Sensitivity (s):", font=("Arial", 10)
        )
        self.sensitivity_label.grid(row=0, column=1)
        self.sensitivity_scale = tk.Scale(
            control_frame, from_=0.1, to=5, resolution=0.1, orient=tk.HORIZONTAL
        )
        self.sensitivity_scale.set(0.5)  # default value
        self.sensitivity_scale.grid(row=1, column=1)

        # Initialize the quit button
        self.quit_button = tk.Button(root, text="Quit", command=self.end_program)
        self.quit_button.pack()
        self.quit_button.config(width=20, height=3)

        # Initialize other variables
        self.thread = None
        self.hold_time = -1.0

    def display_image(self, imgtk, message="changed image"):
        if self.running:
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)
            # print(message)

    def setBackground(self, isBiting):
        if isBiting:
            self.root.configure(background="red")
        else:
            self.root.configure(background="systemWindowBackgroundColor")

    def main_function(self):
        while self.running:
            # Check if the user is biting their nails
            isBiting = bite_detection(self)
            # print(isBiting)

            # Check first bite
            if isBiting and self.hold_time == -1.0:
                self.hold_time = time.time()
            elif not isBiting:
                self.hold_time = -1.0

            # Check if the user has been biting their nails for more than the sensitivity value
            sensitivity = (
                self.sensitivity_scale.get()
            )  # Get the sensitivity value from the slider
            do_catch = isBiting and time.time() - self.hold_time > sensitivity
            if do_catch:
                self.root.focus_force()
            self.setBackground(do_catch)
            time.sleep(0.01)

    def toggle_main_function(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self.main_function)
            self.thread.start()
            self.start_button.config(text="Stop")
        else:
            self.running = False
            self.root.after(100, lambda: stop_thread(self.thread))
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            imgTk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgTk
            self.image_label.configure(image=imgTk)
            self.start_button.config(text="Start")
            self.setBackground(False)
            print("stopped")

    def end_program(self):
        self.running = False
        self.root.after(100, lambda: stop_thread(self.thread))
        self.root.after(200, lambda: self.root.destroy())


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    camera.release()
