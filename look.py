import cv2

# images credits to www.freepik.com and www.vecteezy.com

def _set_look( robot, look_direction="frontal", activated=True):
    if activated:
        path ="./faces/look_"+str(look_direction)+".jpg"
        print(path)
        img = cv2.imread(path)
        image = cv2.resize(img, (1024,600))
        robot._set_display_data(image)