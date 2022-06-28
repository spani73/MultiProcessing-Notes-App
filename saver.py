def work():
    import multiprocessing
    import numpy as np 
    import cv2
    import pyautogui
    import imgcompare
    import time
    import os
    import shutil
    import psutil
    process = "Zoom.exe"
    isthere = True
    src = r"C:\Users\SUBHANKAR\OneDrive\Desktop\Deep-Learning-Face-Recognition-master"
    dest = r"C:\Users\SUBHANKAR\OneDrive\Desktop\Deep-Learning-Face-Recognition-master\dir1"
    count = 1
    while(isthere == True):
        name = multiprocessing.current_process().name
        print(name + " in saver")
        isthere=False
        for proc in psutil.process_iter():
            try:
                processName = proc.name()
                if processName == process:
                    isthere=True
                    print(processName)
            except (psutil.NoSuchProcess , psutil.AccessDenied , psutil.ZombieProcess):
                pass
        
        time.sleep(5)
        image = pyautogui.screenshot()
        image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
        newimage = r"shot" + str(count) + r".jpg"
        cv2.imwrite(newimage,image)
        

        isfilethere=False
        newfile = "shot" + str(count-1) + ".jpg"
        dirs = os.listdir(src)
        for file in dirs:
            if(file == newfile):
                isfilethere = True
                oldimage = "shot" + str(count-1) + ".jpg"
                img1 = cv2.imread(oldimage,0)
                img2 = cv2.imread(newimage,0)
                res  = cv2.absdiff(img1,img2)
                res = res.astype(np.uint8)
                percentage = (np.count_nonzero(res)*100)/res.size

                if percentage>20.0:
                    os.chdir(src)
                    newsrc = src + r"\shot" + str(count-1) + r".jpg"
                    shutil.move(newsrc, dest)
                    count = count + 1 
        if isfilethere == False:
            count=count+1            
                 


    
 
 
 






