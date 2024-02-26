import cv2
import numpy as np
import mediapipe as mp
import math
from cvzone.Utils import overlayPNG
import os
import argparse

# Find distance between two points


def findDistance(p1, p2):

    x1, y1 = p1
    x2, y2 = p2
    distance = math.hypot(x2 - x1, y2 - y1)

    return distance


# Add the Sharingab
def addSharingan(image, centerAxis, radius):
    wh = round(radius * 2)  # Width, Height = radius x 2
    resizedSharingan = cv2.resize(
        sharinganImgList[sharinganImgIndex].copy(), (wh, wh))

    axis = [round(x-radius) for x in centerAxis]

    """
    # Below attempt did not gave the expected result!

    # Cropping (100% - var(irisVisible)) from top so that it looks original
    height, width = resizedSharingan.shape[:2]
    pixels_to_crop = int(height * (1 - irisVisible))
    # Crop the image
    croppedSharingan = resizedSharingan[pixels_to_crop:, :]

    # The image should include the cropped space top height
    axis[1] += pixels_to_crop

    image = overlayPNG(image, croppedSharingan, axis)
    """

    image = overlayPNG(image, resizedSharingan, axis)

    return image


# Detect the Iris of the Eyes and add Sharingan
def detectIrisAndEdit(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    results = FaceMesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return image

    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                            for p in results.multi_face_landmarks[0].landmark])

    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

    center_left = np.array([l_cx, l_cy], dtype=np.int32)
    center_right = np.array([r_cx, r_cy], dtype=np.int32)

    leftTop = mesh_points[LEFT_EYE_TOP]
    leftBottom = mesh_points[LEFT_EYE_BOTTOM]

    rightTop = mesh_points[RIGHT_EYE_TOP]
    rightBottom = mesh_points[RIGHT_EYE_BOTTOM]

    leftDis = findDistance(leftTop, leftBottom)
    rightDis = findDistance(rightTop, rightBottom)

    # radius * 1.5 = 75% height of iris * irisVisible = 70% of 75% iris is the minimum height
    # we are taking left eye's iris as initial height. why? test results says...
    minEyeDis = (l_radius * 1.5) * irisVisible
    # minEyeDisRight = (r_radius * 2) * irisVisible

    # Initially the logic was to make the image size according to the Iris radius * 2 size. But this way the Sharingan Iris became too much big. So, we are giving the Sharingan height as the Eys's top and bottom distance.. Meaning, the space Iris is shown!
    if (leftDis > minEyeDis):
        # image = addSharingan(image, center_left, l_radius)
        # Taking 0.571% as radius
        image = addSharingan(image, center_left, (leftDis/1.75))
    if (rightDis > minEyeDis):
        # image = addSharingan(image, center_right, r_radius)
        # Taking 0.571% as radius
        image = addSharingan(image, center_right, (rightDis/1.75))

    # Change Sharingan version every time Blink happens
    # Can be customized as needed!
    if ((leftDis < minEyeDis) and (rightDis < minEyeDis) and (eyesShown)):
        if (len(sharinganImgList)-1 == sharinganImgIndex):
            globals()["sharinganImgIndex"] = 0
        else:
            globals()["sharinganImgIndex"] = sharinganImgIndex + 1

        globals()["eyesShown"] = False
    elif ((leftDis > minEyeDis) and (rightDis > minEyeDis) and (not eyesShown)):
        globals()["eyesShown"] = True

    return image


# Global and Const variables
# Mediapipe FaceMesh Object
mp_face_mesh = mp.solutions.face_mesh
FaceMesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Eye indices List
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]

# Eyes top and bottom position indices
LEFT_EYE_TOP = 385
LEFT_EYE_BOTTOM = 374

RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# Irises Indices list
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Iris shown %
irisVisible = 0.70  # 70%

# Sharingan image size
sharinganSize = 120

# Get the Sharingan(s)
sharinganImgList = []
sharinganImgIndex = 0

# Folder Paths
# Contains the images which has Black Iris part also
withIris = "./assets/with-iris/"
# Contains the images which only has the sharingan part
withoutIris = "./assets/without-iris/"

for file in os.listdir(withIris):
    if not (file.endswith(".png")):
        continue

    sharinganImg = cv2.imread(f"{withIris}{file}", cv2.IMREAD_UNCHANGED)
    sharinganImg = cv2.resize(sharinganImg, (sharinganSize, sharinganSize))

    sharinganImgList.append(sharinganImg)


# Frame size
FrameWidth, FrameHeight = 640, 360
# Window Name
windowName = "Sharingan"

eyesShown = False
# Live Sharingan from Video


def liveVideoSharingan(sourcePath):
    # Initialize Camera
    cap_vid = cv2.VideoCapture(sourcePath)
    cap_vid.set(cv2.CAP_PROP_FRAME_WIDTH, FrameWidth)
    cap_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, FrameHeight)

    print("\n[*] You are now an Uchiha!")
    print("[*] Press Esc to get back to normal...")
    while True:
        ret, frame = cap_vid.read()
        if not ret:
            break

        frame = detectIrisAndEdit(frame)

        frame = cv2.flip(frame, cv2.CAP_PROP_XI_DECIMATION_HORIZONTAL)
        cv2.imshow(windowName, frame)

        if (cv2.waitKey(1) & 0xFF == 27):
            break
        if (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1):
            break

    # Release camera
    cap_vid.release()


# Sharingan from Image
def staticImageSharingan(sourcePath):
    imgExtn = [".jpg", "jpeg", ".png"]

    if not (any(sourcePath.endswith(x) for x in imgExtn)):
        print("[!] Supported Image formats are: " +
              ", ".join(imgExtn).replace(".", ""))
        exit("\nExiting program...")

    img = cv2.imread(sourcePath, cv2.IMREAD_UNCHANGED)

    print("\n[*] Press Esc to exit or c to change Sharingan Version...")
    while True:
        sharinganEffect = detectIrisAndEdit(img.copy())
        cv2.imshow(windowName, sharinganEffect)
        key = cv2.waitKey(0) & 0xFF
        if (key == 27):
            break
        elif (key == ord("c")):
            # Change Sharingan image when c pressed (c = change)
            if (len(sharinganImgList)-1 == sharinganImgIndex):
                globals()["sharinganImgIndex"] = 0
            else:
                globals()["sharinganImgIndex"] = sharinganImgIndex + 1

        if (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1):
            break


# Creating an Argument Parser
parser = argparse.ArgumentParser(
    description="Sharingan eye Effect from an Image or Video")

parser.add_argument(
    '--source', help="Specify if Image source or Video source - `image` or `video`", default="video")
parser.add_argument(
    '--source-path', help="Specify the Source Path - File / Folder path or Camera Index", default="0")

args = parser.parse_args()

if (args.source == "video"):
    if (args.source_path.isnumeric()):
        sourcePath = int(args.source_path)
    elif not (os.path.isfile(args.source_path)):
        print("\n[!] Source Path must be a Video File or Camera Index")
        exit("\nExiting program...")
    else:
        sourcePath = args.source_path
elif (args.source == "image"):
    if not (os.path.isfile(args.source_path)):
        print("\n[!] Source Path must be an Image File")
        exit("\nExiting program...")
    else:
        sourcePath = args.source_path
else:
    print("\n[!] --source value must be `image` or `video`")


# Start Capturing Process
if (args.source == "video"):
    liveVideoSharingan(sourcePath)
elif (args.source == "image"):
    staticImageSharingan(sourcePath)


# Close all windows
print("\n[*] You are back to being only a Fan!")
cv2.destroyAllWindows()
