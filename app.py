import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Function to overlay an accessory on a frame
def overlay_image(image, overlay, x, y, width, height):
    overlay_resized = cv2.resize(overlay, (width, height))
    for i in range(height):
        for j in range(width):
            if overlay_resized.shape[2] == 4 and overlay_resized[i, j, 3] > 0:  # Check alpha
                if 0 <= y + i < image.shape[0] and 0 <= x + j < image.shape[1]:
                    alpha = overlay_resized[i, j, 3] / 255.0
                    image[y + i, x + j] = (1 - alpha) * image[y + i, x + j] + alpha * overlay_resized[i, j, :3]

# Function for virtual ring try-on
def virtual_try_on_ring():
    st.write("Launching Ring Try-On...")
    cap = cv2.VideoCapture(0)
    ring = cv2.imread('ring.png', cv2.IMREAD_UNCHANGED)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Get the index finger base coordinates
                    x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * frame.shape[1])
                    y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * frame.shape[0])

                    # Adjust the ring position with an offset
                    overlay_image(frame, ring, x - 25, y + 10, 50, 50)

            cv2.imshow("Virtual Ring Try-On", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Function for virtual earring try-on
def virtual_try_on_earring():
    st.write("Launching Earring Try-On...")
    cap = cv2.VideoCapture(0)
    earring = cv2.imread('earring.png', cv2.IMREAD_UNCHANGED)

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # Get coordinates for the left earlobe
                    left_ear_x = int(face_landmarks.landmark[234].x * frame.shape[1])
                    left_ear_y = int(face_landmarks.landmark[234].y * frame.shape[0])
                    # Get coordinates for the right earlobe
                    right_ear_x = int(face_landmarks.landmark[454].x * frame.shape[1])
                    right_ear_y = int(face_landmarks.landmark[454].y * frame.shape[0])

                    # Adjust for better positioning
                    overlay_image(frame, earring, left_ear_x - 25, left_ear_y + 10, 50, 50)
                    overlay_image(frame, earring, right_ear_x - 25, right_ear_y + 10, 50, 50)

            cv2.imshow("Virtual Earring Try-On", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit App
st.title("Ringa-Ring: Virtual Try-On Platform")
st.sidebar.header("Categories")
category = st.sidebar.radio("Select a Category:", ["Rings", "Earrings"])

if category == "Rings":
    st.subheader("Choose Your Favorite Ring")
    col1, col2 = st.columns(2)

    with col1:
        st.image("ring.png", caption="Emerald Ring - $99")
        if st.button("Virtual Try-On", key="ring1"):
            virtual_try_on_ring()

    with col2:
        st.image("ring2.jfif", caption="Heart Shape Ring - $79")
        st.button("Virtual Try-On", key="ring2")  # Placeholder

elif category == "Earrings":
    st.subheader("Choose Your Favorite Earrings")
    col1, col2 = st.columns(2)

    with col1:
        st.image("earring.png", caption="Blue Gem Ring - $49")
        if st.button("Virtual Try-On", key="earring1"):
            virtual_try_on_earring()

    with col2:
        st.image("earring2.jfif", caption="Diamond Earring - $59")
        st.button("Virtual Try-On", key="earring2")  # Placeholder
