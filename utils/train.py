from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"C:\runs\classify\train9\weights\best.pt")
    model.train(data=r'F:\系统默认\桌面\闲鱼\视频异常检测\dataset',epochs=500,device=0,batch=2048)