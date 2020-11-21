import argparse
import cv2
import numpy as np
import torch
import os
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

import time
import threading
import smtplib
# import yagmail
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def send_email(img_path_file = None):
    # yag_server = yagmail.SMTP(user = "a5372935@gmail.com", password = "pivextkynziigmui")
    # email_to = ["jetmaie.fintech@gmail.com",]
    # email_title = "Alarm ~ Fall down"
    # email_content = "The old guy fell down"

    # yag_server.send(email_to, email_title, email_content)
    # yag_server.close()
    with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:  # 設定SMTP伺服器
        try:
            smtp = smtplib.SMTP('smtp.gmail.com', 587)
            smtp.ehlo()
            smtp.starttls()
            smtp.login("a5372935@gmail.com", "pivextkynziigmui") # 登入個人的信箱(應用程式專用密碼)
            from_address = "a5372935@gmail.com"
            to_address = "jetmaie.fintech@gmail.com" # 目標信箱

            
            msg = MIMEMultipart()  #建立MIMEMultipart物件
            msg["subject"] = "Alarm ~ Fall down"  #郵件標題
            msg["from"] = from_address  #寄件者
            msg["to"] = to_address #收件者
            msg.attach(MIMEText("The old guy fell down"))  #郵件內容
            msg.attach(MIMEImage(Path(img_path_file).read_bytes()))
            smtp.send_message(msg)
        except Exception as e:
            print("Error message: ", e)
    # smtp.quit()

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth):
    
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    # pre_key_point = None
    delay = 33
    # 使用 XVID 編碼
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG') # XVID
    # 建立 VideoWriter 物件，輸出影片至 output.avi
    # FPS 值為 20.0，解析度為 640x360
    # out = cv2.VideoWriter('Openpose_demo_fall.avi', fourcc, 30.0, (1280, 720))
    
    for img in image_provider:
        start_time = time.time()
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            Point_dis = track_poses(previous_poses, current_poses, smooth = smooth)
            # print(Point_dis)
            if(Point_dis != None and Point_dis > 25): # 當大於 25 pixel && 警報為 true 執行發送email通知
                if(Pose.Fall_alarm == 1):
                    img_fall_file = "./Fall_img.jpg"
                    cv2.imwrite(img_fall_file, img)
                    Pose.Fall_alarm = -1
                    p = threading.Thread(target=send_email, args = (img_fall_file, )) # 以執行緒同步發送email
                    p.start()
                print(time.time() - Pose.S_time)
                if(time.time() - Pose.S_time > 30):
                    Pose.Fall_alarm = 1
                    Pose.S_time = time.time()
            previous_poses = current_poses
        
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0, img, 1, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        end_time = time.time()
        cv2.putText(img, "FPS : " + str(1 / (end_time - start_time + 1e-8)) , (0, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255))
        # out.write(img)
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        # cv2.waitKey(0)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 33:
                delay = 0
            else:
                delay = 33
        elif key == 113:
            break
    # out.release()

def run_demo_image(net, image_provider, height_size, cpu, track, smooth):
    
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    #print(image_provider[0])
    img = cv2.imread(image_provider[0], cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)
 
    for pose in current_poses:
        pose.draw(img)
    for pose in current_poses:
        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0), 2)
    cv2.namedWindow('Lightweight Human Pose Estimation Python Demo', cv2.WINDOW_NORMAL)
    cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
        run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
    else:
        args.track = 0
        run_demo_image(net, args.images, args.height_size, args.cpu, args.track, args.smooth)

    #run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
