import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io

st.title("Practical 1 - Coordinate Conversion")
st.info("""
INSTRUCTIONS
- Select the grades csv file.
- Select the student submission
- Fill in the student details (ie. student number, surname, and initials)
- Grade by selecting the appropriate checkboxes
- Press submit
- Enter grade in spreadsheet
- Download and save file
""")
gc = st.file_uploader("Select the grades CSV file", type = "csv")
df = pd.read_csv(gc)
surname_list = df["Last Name"].unique()
username_list = df["Username"].unique()
image = st.file_uploader("Select the student submission", type = ["png", "jpg"])
q1_grade = 0
q2_grade = 0
if image is not None:
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv.imdecode(img_bytes, 1)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    parameters =  cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    ids = np.concatenate(ids, axis=0).tolist()
    WIDTH = 712
    HEIGHT = 972
    aruco_top_left = corners[ids.index(0)]
    aruco_top_right = corners[ids.index(1)]
    aruco_bottom_right = corners[ids.index(2)]
    aruco_bottom_left = corners[ids.index(3)]
    point1 = aruco_top_left[0][0]
    point2 = aruco_top_right[0][1]
    point3 = aruco_bottom_right[0][2]
    point4 = aruco_bottom_left[0][3]
    working_image = np.float32([[point1[0], point1[1]],
                                [point2[0], point2[1]],
                                [point3[0], point3[1]],
                                [point4[0], point4[1]]])
    working_target = np.float32([[0, 0],
                                 [WIDTH, 0],
                                 [WIDTH, HEIGHT],
                                 [0, HEIGHT]])
    transformation_matrix = cv.getPerspectiveTransform(working_image, working_target)
    warped_img = cv.warpPerspective(img_gray, transformation_matrix, (WIDTH, HEIGHT))
    details = warped_img[0:280, 0:972]
    q1 = warped_img[285:508, 0:972]
    q2 = warped_img[511:718, 0:972]

    with st.form(key="student_submission"):
        st.image(details)
        snumber_select = st.selectbox("Student Number", options=username_list)
        surname_select = st.selectbox("Surname", options = surname_list)
        initials = st.text_input("Initials")

        st.image(q1)

        chk1a = st.checkbox("Question 1a", key="chk1a")
        chk1b = st.checkbox("Question 1b", key="chk1b")
        chk1c = st.checkbox("Question 1c", key="chk1c")
        chk1d = st.checkbox("Question 1d", key="chk1d")
        chk1e = st.checkbox("Question 1e", key="chk1e")

        q2 = warped_img[511:718, 0:972]
        st.image(q2)

        chk2a = st.checkbox("Question 2a", key="chk2a")
        chk2b = st.checkbox("Question 2b", key="chk2b")
        chk2c = st.checkbox("Question 2c", key="chk2c")
        chk2d = st.checkbox("Question 2d", key="chk2d")
        chk2e = st.checkbox("Question 2e", key="chk2e")


        grade_button = st.form_submit_button()

    if grade_button:
        if st.session_state.chk1a:
            q1_grade += 1
        if st.session_state.chk1b:
            q1_grade += 1
        if st.session_state.chk1c:
            q1_grade += 1
        if st.session_state.chk1d:
            q1_grade += 1
        if st.session_state.chk1e:
            q1_grade += 1
        if st.session_state.chk2a:
            q2_grade += 1
        if st.session_state.chk2b:
            q2_grade += 1
        if st.session_state.chk2c:
            q2_grade += 1
        if st.session_state.chk2d:
            q2_grade += 1
        if st.session_state.chk2e:
            q2_grade += 1

        st.text_input("Q1 Grade", value=q1_grade)
        st.text_input("Q2 Grade", value=q2_grade)

        final_grade = q1_grade + q2_grade
        final_img = cv.putText(img=warped_img, text=f'{q1_grade}', org=(588, 450), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        final_img = cv.putText(img=final_img, text=f'{q2_grade}', org=(588, 640),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        final_img = cv.putText(img=final_img, text=f'{final_grade}', org=(400, 200), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        st.image(final_img)
        filename = f"{surname_select}-{initials}-{snumber_select}.png"
        final_img_rgb = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img)
        buffer = io.BytesIO()
        final_img_pil.save(buffer, format="PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")









