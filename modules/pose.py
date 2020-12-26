import cv2
import numpy as np
import random
from modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from modules.one_euro_filter import OneEuroFilter
import time
# from yolo import YOLO

class Pose:
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [255, 255, 0]
    Fall_alarm = 1
    S_time = time.time()

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img, img_roi_coordinate = None):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                # x_a = x_a + img_roi_coordinate[0]
                # y_a = y_a + img_roi_coordinate[2]
                cv2.circle(img, (int(x_a), int(y_a)), 5, Pose.color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                # x_b = x_b + img_roi_coordinate[0]
                # y_b = y_b + img_roi_coordinate[2]
                # print("kpt_b_id = " + str(Pose.kpt_names[kpt_b_id]) + "; " + str(int(x_b)) + ", " + str(int(y_b)))
                cv2.circle(img, (int(x_b), int(y_b)), 5, Pose.color, -1)
                cv2.putText(img, str(Pose.kpt_names[kpt_b_id]) + " : " + str(int(x_b)) + ", " + str(int(y_b)), 
                (int(x_b), int(y_b)), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 128, 0), 1, cv2.LINE_AA)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3, cv2.LINE_4)


def get_similarity(a, b, threshold = 0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            # print(Pose.kpt_names[kpt_id] + " : " + str(distance))
            # print(distance)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
       
    return num_similar_kpt

def get_distance(current_poses, previous_poses):
    point_distance_id = []
    # print(current_poses[0].keypoints[1, 0])
    # print(len(current_poses))
    for i in range(len(current_poses)):
        try:
            # print("current_poses = " + str(current_poses[i].keypoints[1, 1]))
            # print("previous_poses = " + str(previous_poses[i].keypoints[1, 1]))
            # print("current_poses - previous_poses = " + str(current_poses[i].keypoints[1, 0] - previous_poses[i].keypoints[1, 0]))
            # print("current_poses = " + str(current_poses[0].keypoints[1, 0]))
            # print("previous_poses = " + str(previous_poses[i].keypoints[1, 0]))
            if current_poses[i].keypoints[1, 0] != -1 and previous_poses[i].keypoints[1, 0] != -1:
                
                # calculate variation of two-point
                # x = current_poses[i].keypoints[1][0] - previous_poses[i].keypoints[1][0]
                # y = current_poses[i].keypoints[1][1] - previous_poses[i].keypoints[1][1]
                # point_distance = np.sqrt(x ** 2 + y ** 2)

                # calculate variation of y-axis
                point_distance = previous_poses[i].keypoints[1][1] - current_poses[i].keypoints[1][1]
                point_distance_id.append(point_distance)
            else:
                point_distance_id.append(0)
        except Exception as e:
            pass
        continue
    return point_distance_id

def track_poses(previous_poses, current_poses, threshold = 3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    # print("")
    # print("confidence_before")
    # for i in current_poses:
    #     print(i.confidence)
    current_poses = sorted(current_poses, key = lambda pose: pose.confidence, reverse=True)  # match confident poses first
    # print("confidence_after")
    # for i in current_poses:
    #     print(i.confidence)
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        # print(current_pose.confidence)
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            iou = get_similarity(current_pose, previous_pose)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_id = id        
            
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        
        current_pose.update_id(best_matched_pose_id)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if current_pose.keypoints[kpt_id, 0] == -1:
                    continue
                # reuse filter if previous pose has valid filter
                if (best_matched_pose_id is not None
                        and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                    current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)
        
        # print("current_poses : " + str(len(current_poses)))
        # print("previous_poses : " + str(len(previous_poses)))
    
        if len(previous_poses) != 0: # 當前一個poses位置不為0時
            # Current_neck = current_poses[0].keypoints[1]  # 紀錄neck位置
            # Previous_neck = previous_poses[0].keypoints[1]
            Point_dis = get_distance(current_poses, previous_poses) # 計算 neck 差值
            
            return Point_dis