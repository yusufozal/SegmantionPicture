import cv2
import mediapipe as mp
import numpy as np
mp_selfie_segmentation = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)

BG_COLOR = (69,15,28)
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
     while True:
          ret, frame = cap.read()
          if ret == False:
               break

          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = selfie_segmentation.process(frame_rgb)
          _, th = cv2.threshold(results.segmentation_mask, 0.75, 255, cv2.THRESH_BINARY)
          th=th.astype(np.uint8)
          th=cv2.medianBlur(th,13)
          th_inv=cv2.bitwise_not(th)

          #bg_image=np.ones(frame.shape,dtype=np.uint8)
          #bg_image[:]=BG_COLOR
          bg_image=cv2.imread("dene3.jpeg")
          if frame.shape != bg_image.shape:
              bg_image = np.resize(bg_image, frame.shape)
          assert frame.shape == bg_image.shape

          bg=cv2.bitwise_and(bg_image,bg_image,mask=th_inv)
          fg=cv2.bitwise_and(frame,frame,mask=th)
          out=cv2.add(bg,fg)

          #cv2.imshow("Frame1,1", bg)
          #cv2.imshow("Frame1,2", fg)
          cv2.imshow("Frame1,3", out)
          #cv2.imshow("Frame", frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break

cap.release()
cv2.destroyAllWindows()