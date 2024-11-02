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
- Grade by using the appropriate checkboxes
- Download the graded copy and save in a folder (you will be sending these to the senior tutor)
- Add the grade in the appropriate column on a separate Excel worksheet.
""")
gc = st.file_uploader("Select the grades CSV file", type = "csv")
image = st.file_uploader("Select the student submission", type = ["png", "jpg"])
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
    # get student number
    col1 = warped_img[64:233,62:76]
    col1_blur = cv.blur(col1, (5,5))
    col1_thresh = cv.threshold(col1_blur, 150, 255, cv.THRESH_BINARY_INV)[1]

    col2 = warped_img[64:233,79:93]
    col2_blur = cv.blur(col2, (5, 5))
    col2_thresh = cv.threshold(col2_blur, 150, 255, cv.THRESH_BINARY_INV)[1]

    col3 = warped_img[64:233,96:110]
    col3_blur = cv.blur(col3, (5, 5))
    col3_thresh = cv.threshold(col3_blur, 150, 255, cv.THRESH_BINARY_INV)[1]

    col4 = warped_img[64:233,113:127]
    col4_blur = cv.blur(col4, (5, 5))
    col4_thresh = cv.threshold(col4_blur, 150, 255, cv.THRESH_BINARY_INV)[1]

    col5 = warped_img[64:233,130:144]
    col5_blur = cv.blur(col5, (5, 5))
    col5_thresh = cv.threshold(col5_blur, 150, 255, cv.THRESH_BINARY_INV)[1]

    col6 = warped_img[64:233,147:161]
    col6_blur = cv.blur(col6, (5, 5))
    col6_thresh = cv.threshold(col6_blur, 150, 255, cv.THRESH_BINARY_INV)[1]

    col7 = warped_img[64:233,164:178]
    col7_blur = cv.blur(col7, (5, 5))
    col7_thresh = cv.threshold(col7_blur, 150, 255, cv.THRESH_BINARY_INV)[1]

    col8 = warped_img[64:233,181:195]
    col8_blur = cv.blur(col8, (5, 5))
    col8_thresh = cv.threshold(col8_blur, 150, 255, cv.THRESH_BINARY_INV)[1]

    col_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    col1_vals = []
    col1_vals.append(col1[0:17].mean())  # 1
    col1_vals.append(col1[18:34].mean())  # 2
    col1_vals.append(col1[35:51].mean())  # 3
    col1_vals.append(col1[52:68].mean())  # 4
    col1_vals.append(col1[69:85].mean())  # 5
    col1_vals.append(col1[86:102].mean())  # 6
    col1_vals.append(col1[103:119].mean())  # 7
    col1_vals.append(col1[120:136].mean())  # 8
    col1_vals.append(col1[137:153].mean())  # 9
    col1_vals.append(col1[154:170].mean())  # 0
    col1_min = min(col1_vals)
    col1_index = col1_vals.index(col1_min)
    digit1 = col_vals[col1_index]

    col2_vals = []
    col2_vals.append(col2[0:17].mean())  # 1
    col2_vals.append(col2[18:34].mean())  # 2
    col2_vals.append(col2[35:51].mean())  # 3
    col2_vals.append(col2[52:68].mean())  # 4
    col2_vals.append(col2[69:85].mean())  # 5
    col2_vals.append(col2[86:102].mean())  # 6
    col2_vals.append(col2[103:119].mean())  # 7
    col2_vals.append(col2[120:136].mean())  # 8
    col2_vals.append(col2[137:153].mean())  # 9
    col2_vals.append(col2[154:170].mean())  # 0
    col2_min = min(col2_vals)
    col2_index = col2_vals.index(col2_min)
    digit2 = col_vals[col2_index]

    col3_vals = []
    col3_vals.append(col3[0:17].mean())  # 1
    col3_vals.append(col3[18:34].mean())  # 2
    col3_vals.append(col3[35:51].mean())  # 3
    col3_vals.append(col3[52:68].mean())  # 4
    col3_vals.append(col3[69:85].mean())  # 5
    col3_vals.append(col3[86:102].mean())  # 6
    col3_vals.append(col3[103:119].mean())  # 7
    col3_vals.append(col3[120:136].mean())  # 8
    col3_vals.append(col3[137:153].mean())  # 9
    col3_vals.append(col3[154:170].mean())  # 0
    col3_min = min(col3_vals)
    col3_index = col3_vals.index(col3_min)
    digit3 = col_vals[col3_index]

    col4_vals = []
    col4_vals.append(col4[0:17].mean())  # 1
    col4_vals.append(col4[18:34].mean())  # 2
    col4_vals.append(col4[35:51].mean())  # 3
    col4_vals.append(col4[52:68].mean())  # 4
    col4_vals.append(col4[69:85].mean())  # 5
    col4_vals.append(col4[86:102].mean())  # 6
    col4_vals.append(col4[103:119].mean())  # 7
    col4_vals.append(col4[120:136].mean())  # 8
    col4_vals.append(col4[137:153].mean())  # 9
    col4_vals.append(col4[154:170].mean())  # 0
    col4_min = min(col4_vals)
    col4_index = col4_vals.index(col4_min)
    digit4 = col_vals[col4_index]

    col5_vals = []
    col5_vals.append(col5[0:17].mean())  # 1
    col5_vals.append(col5[18:34].mean())  # 2
    col5_vals.append(col5[35:51].mean())  # 3
    col5_vals.append(col5[52:68].mean())  # 4
    col5_vals.append(col5[69:85].mean())  # 5
    col5_vals.append(col5[86:102].mean())  # 6
    col5_vals.append(col5[103:119].mean())  # 7
    col5_vals.append(col5[120:136].mean())  # 8
    col5_vals.append(col5[137:153].mean())  # 9
    col5_vals.append(col5[154:170].mean())  # 0
    col5_min = min(col5_vals)
    col5_index = col5_vals.index(col5_min)
    digit5 = col_vals[col5_index]

    col6_vals = []
    col6_vals.append(col6[0:17].mean())  # 1
    col6_vals.append(col6[18:34].mean())  # 2
    col6_vals.append(col6[35:51].mean())  # 3
    col6_vals.append(col6[52:68].mean())  # 4
    col6_vals.append(col6[69:85].mean())  # 5
    col6_vals.append(col6[86:102].mean())  # 6
    col6_vals.append(col6[103:119].mean())  # 7
    col6_vals.append(col6[120:136].mean())  # 8
    col6_vals.append(col6[137:153].mean())  # 9
    col6_vals.append(col6[154:170].mean())  # 0
    col6_min = min(col6_vals)
    col6_index = col6_vals.index(col6_min)
    digit6 = col_vals[col6_index]

    col7_vals = []
    col7_vals.append(col7[0:17].mean())  # 1
    col7_vals.append(col7[18:34].mean())  # 2
    col7_vals.append(col7[35:51].mean())  # 3
    col7_vals.append(col7[52:68].mean())  # 4
    col7_vals.append(col7[69:85].mean())  # 5
    col7_vals.append(col7[86:102].mean())  # 6
    col7_vals.append(col7[103:119].mean())  # 7
    col7_vals.append(col7[120:136].mean())  # 8
    col7_vals.append(col7[137:153].mean())  # 9
    col7_vals.append(col7[154:170].mean())  # 0
    col7_min = min(col7_vals)
    col7_index = col7_vals.index(col7_min)
    digit7 = col_vals[col7_index]

    col8_vals = []
    col8_vals.append(col8[0:17].mean())  # 1
    col8_vals.append(col8[18:34].mean())  # 2
    col8_vals.append(col8[35:51].mean())  # 3
    col8_vals.append(col8[52:68].mean())  # 4
    col8_vals.append(col8[69:85].mean())  # 5
    col8_vals.append(col8[86:102].mean())  # 6
    col8_vals.append(col8[103:119].mean())  # 7
    col8_vals.append(col8[120:136].mean())  # 8
    col8_vals.append(col8[137:153].mean())  # 9
    col8_vals.append(col8[154:170].mean())  # 0
    col8_min = min(col8_vals)
    col8_index = col8_vals.index(col8_min)
    digit8 = col_vals[col8_index]

    snumber = f'u{str(digit1)}{str(digit2)}{str(digit3)}{str(digit4)}{str(digit5)}{str(digit6)}{str(digit7)}{str(digit8)}'
    snumber_txt = st.text_input("**Student Number**", value=snumber)
    df = pd.read_csv(gc)
    row_index = df.index[df["Username"] == snumber].tolist()
    surname = df.iloc[row_index, 0].values[0]
    first = df.iloc[row_index, 1].values[0]
    st.text_input("**Surname**", value=surname)
    st.text_input("**Initials**", value=first)

    # get regions of interest
    details = warped_img[0:280, 0:972]
    st.image(details)
    q1 = warped_img[285:508, 0:972]
    st.image(q1)

    chk1a = st.checkbox("Question 1a: -22.35139", key="chk1a")
    chk1b = st.checkbox("Question 1b: +30.03972", key="chk1b")
    chk1c = st.checkbox("Question 1c: +30.04444", key="chk1c")
    chk1d = st.checkbox("Question 1d: -31.23583", key="chk1d")
    chk1e = st.checkbox("Question 1e: -15.79750", key="chk1e")

    q2 = warped_img[511:718, 0:972]
    st.image(q2)

    chk2a = st.checkbox("Question 2a: 17°36'30''N", key="chk2a")
    chk2b = st.checkbox("Question 2b: 8°04'54''E", key="chk2b")
    chk2c = st.checkbox("Question 2c: 64°50'24''N", key="chk2c")
    chk2d = st.checkbox("Question 2d: 147°43'12''W", key="chk2d")
    chk2e = st.checkbox("Question 2e: 33°55'24''S", key="chk2e")

    q1_grade = 0
    q2_grade = 0
    if st.button("Grade"):
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
        st.text_input("Q2 Grade", value = q2_grade)

        final_grade = q1_grade + q2_grade
        final_img = cv.putText(img=warped_img, text=f'{q1_grade}', org=(588, 450), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        final_img = cv.putText(img=final_img, text=f'{q2_grade}', org=(588, 640),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        final_img = cv.putText(img=final_img, text=f'{final_grade}', org=(400, 200), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        st.image(final_img)
        filename = f"{surname}-{first}-{snumber}.png"
        final_img_rgb = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img)
        buffer = io.BytesIO()
        final_img_pil.save(buffer, format="PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")
        #df.iloc[row_index, 6] = final_grade
        #csv_download = df.to_csv(index=False)
        #st.download_button(label = "Download Grade", data=csv_download, file_name=gc.name, mime="text/csv")







