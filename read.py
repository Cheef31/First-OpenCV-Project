import cv2 as cv2

# Video lesen; 0 = Webcam; 1 = Iriun Webcam
cap = cv2.VideoCapture(1)

# Object detection
object_detector = cv2.createBackgroundSubtractorMOG2()

# Video anzeigen
while True:
    ret, frame = cap.read()
    
    # ROI
    # roi=frame[y1:y2,x1:x2]
    roi=frame[100:220,0:640]

    # Object Detektion
    mask = object_detector.apply(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calculate area und kleine Elemente entfernen
        area = cv2.contourArea(cnt)
        if area > 50:
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)

    #cv2.imshow('Video', frame)
    cv2.imshow('ROI', roi)

    key = cv2.waitKey(30)

    # Falls ESC gedrückt wird -> break
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()