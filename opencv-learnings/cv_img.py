import cv2

img = cv2.imread("../img/vicky.jpg")
while True:
    cv2.imshow('vicky', img)

    # If we have waited at least 1 ms and the esc button is pressed, then exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()