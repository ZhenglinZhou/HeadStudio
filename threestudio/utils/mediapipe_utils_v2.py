import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection  # Only for counting faces.
mp_face_mesh = mp.solutions.face_mesh
mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION

DrawingSpec = mp.solutions.drawing_styles.DrawingSpec

# EMOCA-FLAME2MediaPipe-105
right_eyebrow = [
    (9, 6), (6, 8), (8, 4), (4, 7),  # up
    (3, 5), (5, 1), (1, 2), (2, 0),  # down
]

left_eyebrow = [
    (17, 14), (14, 18), (18, 16), (16, 19),  # up
    (10, 12), (12, 11), (11, 15), (15, 13),  # down
]

right_eye = [
    (22, 34), (34, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 35), (35, 21),  # up
    (22, 27), (27, 26), (26, 25), (25, 24), (24, 23), (23, 33), (33, 20), (20, 21),  # down
]

left_eye = [
    (37, 51), (51, 48), (48, 47), (47, 46), (46, 45), (45, 44), (44, 50), (50, 38),  # up
    (37, 36), (36, 49), (49, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 38),  # down
]

lips = [
    (72, 85), (85, 71), (71, 70), (70, 69), (69, 65), (65, 87), (87, 88), (88, 89), (89, 103), (103, 90),  # up1
    (72, 82), (82, 80), (80, 84), (84, 77), (77, 68), (68, 95), (95, 102), (102, 98), (98, 100), (100, 90),  # down1
    (73, 86), (86, 74), (74, 75), (75, 76), (76, 66), (66, 94), (94, 93), (93, 92), (92, 104), (104, 91),  # up2
    (73, 81), (81, 79), (79, 83), (83, 78), (78, 67), (67, 96), (96, 101), (101, 97), (97, 99), (99, 91),  # down2
]
ConnectionDict = {
    'right_eye': right_eye,
    'left_eye': left_eye,
    'right_eyebrow': right_eyebrow,
    'left_eyebrow': left_eyebrow,
    'lips': lips,
}

f_thick = 2
f_rad = 1

right_eye_draw = DrawingSpec(
    color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad
)
right_eyebrow_draw = DrawingSpec(
    color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad
)
left_eye_draw = DrawingSpec(
    color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad
)
left_eyebrow_draw = DrawingSpec(
    color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad
)
mouth_draw = DrawingSpec(
    color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad
)
head_draw = DrawingSpec(
    color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad
)
ColorDict = {
    'right_eye': right_eye_draw,
    'left_eye': left_eye_draw,
    'right_eyebrow': right_eyebrow_draw,
    'left_eyebrow': left_eyebrow_draw,
    'lips': mouth_draw,
}


def draw_landmarks_105(image, landmark_list, connections=ConnectionDict):
    '''
        image: (bs, H, W, 3)
        landmark_list: (bs, num_lmk, 2)
    '''
    bs = image.shape[0]

    def draw_landmarks(img, lmks):
        for key, value in connections.items():
            drawing_spec = ColorDict[key]
            for (start_idx, end_idx) in value:
                cv2.line(
                    img,
                    lmks[start_idx].astype(int),
                    lmks[end_idx].astype(int),
                    drawing_spec.color,
                    drawing_spec.thickness,
                )

        return img[:, :, ::-1]  # flip BGR

    draw_image = np.stack([draw_landmarks(image[i], landmark_list[i])
                           for i in range(bs)], axis=0)
    return draw_image
