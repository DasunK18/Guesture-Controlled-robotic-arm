import serial
import cv2
import mediapipe as mp

# config
write_video = True
debug = True
if not debug:
    ser = serial.Serial('COM4', 115200)

# Set the cam_source to the IP address provided by DroidCam (replace with your actual IP)
cam_source = "http://100.79.186.121:4747/video" # Replace with your phone's IP address

# x-axis control range
x_min = 0
x_mid = 75
x_max = 150
palm_angle_min = -50
palm_angle_mid = 20

servo_angle = [x_mid]

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
map_range = lambda x, in_min, in_max, out_min, out_max: abs(
    (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)


# Check if the hand is a fist
def is_fist(hand_landmarks, palm_size):
    distance_sum = 0
    WRIST = hand_landmarks.landmark[0]
    for i in [7, 8, 11, 12, 15, 16, 19, 20]:
        distance_sum += ((WRIST.x - hand_landmarks.landmark[i].x) ** 2 + \
                         (WRIST.y - hand_landmarks.landmark[i].y) ** 2 + \
                         (WRIST.z - hand_landmarks.landmark[i].z) ** 2) ** 0.5
    return distance_sum / palm_size < 7  # fist threshold


def landmark_to_servo_angle(hand_landmarks):
    servo_angle = [x_mid]  # Only control x-axis servo, initialize to mid value
    WRIST = hand_landmarks.landmark[0]
    INDEX_FINGER_MCP = hand_landmarks.landmark[5]

    # Calculate palm size (distance between wrist and index finger)
    palm_size = ((WRIST.x - INDEX_FINGER_MCP.x) ** 2 + (WRIST.y - INDEX_FINGER_MCP.y) ** 2 + (
                WRIST.z - INDEX_FINGER_MCP.z) ** 2) ** 0.5

    # Calculate x angle (between wrist and index finger)
    distance = palm_size
    angle = (WRIST.x - INDEX_FINGER_MCP.x) / distance  # Calculate the radian between wrist and index finger
    angle = int(angle * 180 / 3.1415926)  # Convert radian to degree
    angle = clamp(angle, palm_angle_min, palm_angle_mid)
    servo_angle[0] = map_range(angle, palm_angle_min, palm_angle_mid, x_max, x_min)

    # Return only the x-axis servo angle
    return servo_angle


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(cam_source)

# Set the resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



# video writer
if write_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process the image
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 1:
                hand_landmarks = results.multi_hand_landmarks[0]
                servo_angle = landmark_to_servo_angle(hand_landmarks)

                # Check if servo angle has changed
                print("Servo angle (x-axis): ", servo_angle[0])

                # Send the servo angle to the serial port (if not in debug mode)
                if not debug:
                    ser.write(bytearray([servo_angle[0]]))  # Send only the x-axis servo angle
            else:
                print("More than one hand detected")

            # Draw landmarks and connections
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)

        # Show the servo angle on the screen
        cv2.putText(image, f"Servo x-angle: {servo_angle[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow('MediaPipe Hands', image)

        if write_video:
            out.write(image)

        # Exit on 'Esc' key press
        if cv2.waitKey(5) & 0xFF == 27:
            if write_video:
                out.release()
            break

cap.release()
