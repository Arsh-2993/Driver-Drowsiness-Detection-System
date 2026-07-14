import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from tensorflow.keras.models import load_model
import pygame
import customtkinter as ctk
from PIL import Image
import sqlite3
import matplotlib.pyplot as plt


base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "drowsiness_model.h5")
predictor_path = os.path.join(base_path, "modelsshape_predictor_68_face_landmarks.dat")
alarm_path = os.path.join(base_path, "alarm.wav")


conn = sqlite3.connect("users.db")
cursor = conn.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
cursor.execute("CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, yawns INTEGER, drowsy INTEGER)")
conn.commit()


model = load_model(model_path)
labels = ['Closed', 'Open', 'no_yawn', 'yawn']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

pygame.mixer.init()
pygame.mixer.music.load(alarm_path)
pygame.mixer.music.set_volume(1.0)


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)


cam_running = False
alarm_on = False
frame_count = 0
yawn_count = 0
cap = None


def update_camera():
    global frame_count, yawn_count, alarm_on

    if not cam_running:
        return

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_camera)
        return

    frame = cv2.resize(frame, (600, 380))
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

        EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        MAR = mouth_aspect_ratio(mouth)

        
        label = "Unknown"
        try:
            face_img = gray[face.top():face.bottom(), face.left():face.right()]
            face_img = cv2.resize(face_img, (64, 64))
            face_img = face_img / 255.0
            face_img = face_img.reshape(1, 64, 64, 1)
            pred = model.predict(face_img, verbose=0)
            label = labels[np.argmax(pred)]
        except:
            pass

        if EAR < 0.25:
            frame_count += 1
            if frame_count > 10:
                drowsy_status.configure(text="Drowsy", text_color="red")
                if not alarm_on:
                    pygame.mixer.music.play(-1)
                    alarm_on = True
        else:
            frame_count = 0
            drowsy_status.configure(text="Normal", text_color="green")
            if alarm_on:
                pygame.mixer.music.stop()
                alarm_on = False

        
        if MAR > 0.5:
            yawn_count += 1
        else:
            yawn_count = max(0, yawn_count - 1)

        if yawn_count > 5:
            yawn_status.configure(text="Yawning", text_color="red")
        else:
            yawn_status.configure(text="No", text_color="green")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ctk.CTkImage(light_image=img, size=(600, 380))
    camera_label.configure(image=imgtk)
    camera_label.imgtk = imgtk

    root.after(10, update_camera)


def start_cam():
    global cam_running, cap
    clear_main()
    build_camera_ui()
    cap = cv2.VideoCapture(0)
    cam_running = True
    update_camera()


def stop_cam():
    global cam_running
    cam_running = False
    pygame.mixer.music.stop()
    cursor.execute("INSERT INTO sessions (yawns, drowsy) VALUES (?,?)", (yawn_count, frame_count))
    conn.commit()


def clear_main():
    for widget in main.winfo_children():
        widget.destroy()


def show_dashboard():
    clear_main()

    ctk.CTkLabel(main, text="Dashboard", font=("Arial", 26, "bold")).pack(pady=20)

    ctk.CTkLabel(main, text=f"Yawns: {yawn_count}", font=("Arial", 18)).pack(pady=10)
    ctk.CTkLabel(main, text=f"Drowsy Events: {frame_count}", font=("Arial", 18)).pack(pady=10)


def show_chart():
    data = [yawn_count, frame_count]
    labels = ['Yawns', 'Drowsy']

    plt.figure()
    plt.bar(labels, data)
    plt.title("Driver Analytics")
    plt.show()


def build_camera_ui():
    global camera_label, yawn_status, drowsy_status

    camera_label = ctk.CTkLabel(main, text="")
    camera_label.pack(pady=10)

    yawn_status = ctk.CTkLabel(main, text="No", text_color="green")
    yawn_status.pack()

    drowsy_status = ctk.CTkLabel(main, text="Normal", text_color="green")
    drowsy_status.pack()


def login():
    user = username.get()
    pwd = password.get()

    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
    if cursor.fetchone():
        login_frame.pack_forget()
        app_frame.pack(fill="both", expand=True)
        show_dashboard()
    else:
        error_label.configure(text="Invalid Credentials")


def signup():
    user = username.get()
    pwd = password.get()
    try:
        cursor.execute("INSERT INTO users VALUES (?,?)", (user, pwd))
        conn.commit()
        error_label.configure(text="Account Created", text_color="green")
    except:
        error_label.configure(text="User Exists", text_color="red")


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.geometry("1100x650")
root.title("Driver Drowsiness Detecting")


login_frame = ctk.CTkFrame(root)
login_frame.pack(fill="both", expand=True)

ctk.CTkLabel(login_frame, text=" Driver Drowsiness Detecting", font=("Arial", 28, "bold")).pack(pady=20)

username = ctk.CTkEntry(login_frame, placeholder_text="Username", width=250)
username.pack(pady=10)

password = ctk.CTkEntry(login_frame, placeholder_text="Password", show="*", width=250)
password.pack(pady=10)

error_label = ctk.CTkLabel(login_frame, text="")
error_label.pack()

ctk.CTkButton(login_frame, text="Login", command=login).pack(pady=10)
ctk.CTkButton(login_frame, text="Sign Up", command=signup).pack(pady=5)


app_frame = ctk.CTkFrame(root)

sidebar = ctk.CTkFrame(app_frame, width=200)
sidebar.pack(side="left", fill="y")

ctk.CTkButton(sidebar, text="Dashboard", command=show_dashboard).pack(pady=10)
ctk.CTkButton(sidebar, text="Live Monitor", command=start_cam).pack(pady=10)
ctk.CTkButton(sidebar, text="Analytics", command=show_chart).pack(pady=10)
ctk.CTkButton(sidebar, text="Stop", command=stop_cam).pack(pady=10)

main = ctk.CTkFrame(app_frame)
main.pack(side="right", expand=True, fill="both")

root.mainloop()
