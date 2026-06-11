import cv2
import os


def augment_image(input_path, output_folder):

    image = cv2.imread(input_path)

    cv2.imwrite(
        os.path.join(output_folder, "original.jpg"),
        image
    )

    flip = cv2.flip(image, 1)
    cv2.imwrite(
        os.path.join(output_folder, "flip.jpg"),
        flip
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    cv2.imwrite(
        os.path.join(output_folder, "gray.jpg"),
        gray
    )

    bright = cv2.convertScaleAbs(
        image,
        alpha=1,
        beta=50
    )

    cv2.imwrite(
        os.path.join(output_folder, "bright.jpg"),
        bright
    )

    dark = cv2.convertScaleAbs(
        image,
        alpha=1,
        beta=-50
    )

    cv2.imwrite(
        os.path.join(output_folder, "dark.jpg"),
        dark
    )

    h, w = image.shape[:2]

    M1 = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        10,
        1
    )

    left = cv2.warpAffine(
        image,
        M1,
        (w, h)
    )

    cv2.imwrite(
        os.path.join(output_folder,
                     "rotate_left.jpg"),
        left
    )

    M2 = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        -10,
        1
    )

    right = cv2.warpAffine(
        image,
        M2,
        (w, h)
    )

    cv2.imwrite(
        os.path.join(output_folder,
                     "rotate_right.jpg"),
        right
    )