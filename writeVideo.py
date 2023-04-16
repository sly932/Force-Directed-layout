import cv2
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import os
import argparse
import time
BLOCK = '█'
BLOCKS_NUM = 50
def printprogress(cur,total,total_sec,epoch_sec,block=BLOCK,progress_len = BLOCKS_NUM):
        hour = int(total_sec // 3600)
        minute = int(total_sec // 60 % 60)
        second = int(total_sec % 60)
        progress = float(cur + 1) / total * 100
        block_num = int(progress * progress_len / 100)
        bar = block * block_num + ' ' * (progress_len - block_num) + '|'
        print(f"\rProgressing:{cur}|{total}|{bar}{progress:.1f}% {hour:02d}:{minute:02d}:{second:02d} {epoch_sec:.2f}sec/epoch",end=' ')

"""def svgToJPG():
    files = [f for f in os.listdir('./') if f.endswith('.svg')]
    total = len(files)
    for i, file in enumerate(files):
        printprogress(i,total)
        renderPM.drawToFile(svg2rlg(file),file.split('.')[0]+'.jpg',fmt='JPG')"""

def writeVideo(args):
    
    path = os.path.join('imgs',args.dir,'iter')
    files = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    files.sort()
    h,w,_ = cv2.imread(files[0]).shape
    k = float(h) / w
    print(f"shape:{w}x{h}")
    w= min(640,w)
    h = int(w * k)
    print(f"shape:{w}x{h}")
    vname = f"{args.name}_fps_{args.fps}.mp4"
    print(f"making video of fps {args.fps}:")
    video = cv2.VideoWriter(vname,cv2.VideoWriter_fourcc('m','p','4','v'),args.fps,(w,h))
    start_time = time.time()
    pre_time = start_time
    for i,file in enumerate(files):
        cur_time = time.time()
        printprogress(i,len(files),cur_time-start_time,cur_time-pre_time)
        pre_time = cur_time
        img = cv2.imread(file)
        img = cv2.resize(img,(w,h))
        video.write(img)
    video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="write video")
    parser.add_argument("--dir",type=str,default=None)
    parser.add_argument("--name",type=str,default="noname")
    parser.add_argument("--fps",type=int,default=120)
    args = parser.parse_args()
    writeVideo(args)