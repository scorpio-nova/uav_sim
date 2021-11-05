#!/usr/bin/python
# -*- encoding: utf8 -*-

# 对windows.world的一个简单控制策略
# 结合tello的控制接口，控制无人机从指定位置起飞，识别模拟火情标记（红色），穿过其下方对应的窗户，并在指定位置降落
# 本策略尽量使无人机的偏航角保持在初始值（90度）左右
# 运行roslaunch uav_sim windows.launch后，再在另一个终端中运行rostopic pub /tello/cmd_start std_msgs/Bool "data: 1"即可开始飞行
# 代码中的decision()函数和switchNavigatingState()函数共有3个空缺之处，需要同学们自行补全（每个空缺之处需要填上不超过3行代码）
# path2: 3-1-4-5

from scipy.spatial.transform import Rotation as R
from collections import deque
from enum import Enum
import rospy
import cv2
import numpy as np
import math
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ControllerNode:
    class FlightState(Enum):  # 飞行状态
        WAITING = 1
        NAVIGATING = 2
        DETECTING_TARGET = 3
        LANDING = 4
        NAVIGATING1 = 5
        NAVIGATING2 = 6
        NAVIGATING3 = 7
        NAVIGATING4 = 8
        NAVIGATING_FINAL = 9

    def __init__(self):
        rospy.init_node('controller_node', anonymous=True)
        rospy.logwarn('Controller node set up.')

        # 无人机在世界坐标系下的位姿
        self.R_wu_ = R.from_quat([0, 0, 0, 1])
        self.t_wu_ = np.zeros([3], dtype=np.float64)

        self.image_ = None
        self.color_range_red = [(0, 43, 46), (6, 255, 255)]  # 红色的HSV范围
        self.color_range_yellow = [(26, 43, 46), (34, 255, 255)]  # 黄色的HSV范围
        self.color_range_blue = [(100, 43, 46), (124, 255, 255)]  # 蓝色的HSV范围
        self.bridge_ = CvBridge()

        self.flight_state_ = self.FlightState.WAITING
        self.navigating_queue_ = deque()  # 存放多段导航信息的队列，队列元素为二元list，list的第一个元素代表导航维度（'x' or 'y' or 'z'），第二个元素代表导航目的地在该维度的坐标
        self.navigating_dimension_ = None  # 'x' or 'y' or 'z'
        self.navigating_destination_ = None
        self.next_state_ = None  # 完成多段导航后将切换的飞行状态

        self.window_x_list_ = [1.75, 4.25, 6.75]  # 窗户中心点对应的x值
        # sphere's (x,y,z)
        self.sphere1 = [6.5, 7, 1.72]
        self.sphere2 = [3.5, 7.5, 0.72]
        self.sphere3 = [5, 9.5, 1]
        self.sphere4 = [4, 11, 1.72]
        self.sphere5 = [1, 14.5, 0.2]
        # shift
        x_shift = [1, 0, 0]
        y_shift = [0, 1, 0]
        # target position
        self.target1 = [self.sphere3[i] - 1.25 * y_shift[i] for i in range(3)]  # 球3 西侧
        self.target2 = [self.sphere1[i] + 1.25 * y_shift[i] for i in range(3)]  # 球1 东侧
        self.target3 = [self.sphere4[i] + 1.25 * y_shift[i] for i in range(3)]  # 球4 东侧
        self.target4 = [self.sphere5[i] + 1.25 * x_shift[i] for i in range(3)]  # 球5 南侧
        self.target_yaw = 90  # 初始化为识别window红点时的方向
        self.target_pitch = 0
        self.yaw_x = None
        self.yaw_y = None
        self.adjust_yaw_pos = False
        self.stage = 0

        # 超参数
        self.min_navigation_distance = 0.2  # 最小导航距离：如果当前无人机位置与目标位置在某个轴方向距离小于这个值，即不再这个轴方向上运动
        self.commandlist = ['e', 'e', 'e', 'e', 'e']
        # 无人机通讯相关
        self.is_begin_ = False
        self.commandPub_ = rospy.Publisher('/tello/cmd_string', String, queue_size=100)  # 发布tello格式控制信号
        self.commandResult_ = rospy.Publisher('/tello/target_result', String, queue_size=5) # 发布目标识别、检测的结果
        self.poseSub_ = rospy.Subscriber('/tello/states', PoseStamped, self.poseCallback)  # 接收处理含噪无人机位姿信息
        self.imageSub_ = rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.imageCallback)  # 接收摄像头图像
        self.imageSub_ = rospy.Subscriber('/tello/cmd_start', Bool, self.startcommandCallback)  # 接收开始飞行的命令
        rate = rospy.Rate(0.3)

        while not rospy.is_shutdown():
            if self.is_begin_:
                self.decision()
            rate.sleep()
        rospy.logwarn('Controller node shut down.')

    # 按照一定频率进行决策，并发布tello格式控制信号
    def decision(self):
        # 起飞
        if self.flight_state_ == self.FlightState.WAITING:  # 起飞并飞至离墙体（y = 3.0m）适当距离的位置
            rospy.logwarn('State: WAITING')
            self.publishCommand('takeoff')
            self.navigating_queue_ = deque([['y', 1.8]])
            self.switchNavigatingState()
            self.next_state_ = self.FlightState.DETECTING_TARGET
        # 巡航
        elif self.flight_state_ == self.FlightState.NAVIGATING:
            rospy.logwarn('State: NAVIGATING')
            rospy.loginfo('current position (x,y,z) = (%.2f,%.2f,%.2f)' % (self.t_wu_[0], self.t_wu_[1], self.t_wu_[2]))
            # 如果yaw与90度相差超过正负10度，需要进行旋转调整yaw
            self.adjust_yaw(self.target_yaw)
            # self.adjust_pitch(self.target_pitch)
            dim_index = 0 if self.navigating_dimension_ == 'x' else (1 if self.navigating_dimension_ == 'y' else 2)
            dist = self.navigating_destination_ - self.t_wu_[dim_index]
            if abs(dist) < self.min_navigation_distance:  # 当前段导航结束
                self.switchNavigatingState()
            else:
                dir_index = 0 if dist > 0 else 1  # direction index
                # TODO 2: 根据维度（dim_index）和导航方向（dir_index）决定使用哪个命令
                if self.target_yaw == 90:
                    command_matrix = [['right ', 'left '], ['forward ', 'back '], ['up ', 'down ']]
                elif self.target_yaw == 0:
                    command_matrix = [['forward ', 'back '], ['left ', 'right '], ['up ', 'down ']]
                elif self.target_yaw == -90:
                    command_matrix = [['left ', 'right '], ['back ', 'forward '], ['up ', 'down ']]
                else: # self.target_yaw == -180:
                    command_matrix = [['back ', 'forward '], ['right ', 'left '], ['up ', 'down ']]
                command = command_matrix[dim_index][dir_index]
                # end of TODO 2
                if abs(dist) > 1.5:
                    self.publishCommand(command + '100')
                else:
                    self.publishCommand(command + str(int(abs(100 * dist))))

        elif self.flight_state_ == self.FlightState.DETECTING_TARGET:
            rospy.logwarn('State: DETECTING_TARGET')
            # 如果无人机飞行高度与标识高度（1.75m）相差太多，则需要进行调整
            if self.t_wu_[2] > 2.0:
                self.publishCommand('down %d' % int(100 * (self.t_wu_[2] - 1.75)))
                return
            elif self.t_wu_[2] < 1.5:
                self.publishCommand('up %d' % int(-100 * (self.t_wu_[2] - 1.75)))
                return
            # 如果yaw与90度相差超过正负10度，需要进行旋转调整yaw
            (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
            yaw_diff = yaw - 90 if yaw > -90 else yaw + 270
            if yaw_diff > 10:  # clockwise
                self.publishCommand('cw %d' % (int(yaw_diff) if yaw_diff > 15 else 15))
                return
            elif yaw_diff < -10:  # counterclockwise
                self.publishCommand('ccw %d' % (int(-yaw_diff) if yaw_diff < -15 else 15))
                return

            if self.detectTarget():
                # rospy.loginfo('Target detected.')
                # 根据无人机当前x坐标判断正确的窗口是哪一个
                # 实际上可以结合目标在图像中的位置和相机内外参数得到标记点较准确的坐标，这需要相机成像的相关知识
                # 此处仅仅是做了一个粗糙的估计
                win_dist = [abs(self.t_wu_[0] - win_x) for win_x in self.window_x_list_]
                win_index = win_dist.index(min(win_dist))  # 正确的窗户编号
                rospy.loginfo('Go to window (0-2): %d' % win_index)
                self.navigating_queue_ = deque(
                    [['y', 2.4], ['z', 1.0], ['x', self.window_x_list_[win_index]],
                     ['y', 4.0], ['x', 3]])  # 通过窗户并导航至特定位置
                self.switchNavigatingState()
                self.next_state_ = self.FlightState.NAVIGATING1
            else:
                if self.t_wu_[0] > 7.5:
                    rospy.loginfo('Detection failed, ready to land.')
                    self.flight_state_ = self.FlightState.LANDING
                else:  # 向右侧平移一段距离，继续检测
                    self.publishCommand('right 75')

        elif self.flight_state_ == self.FlightState.NAVIGATING1:
            if self.stage == 0:  # 导航与检测
                rospy.loginfo('***NAVIGATING1(win->3)...***')
                rospy.loginfo('[[Stage 0 - goto 3]]')
                self.navigating_queue_ = deque(
                    [['y', 6.0], ['x', 5.0], ['y', self.target1[1]], ['x', self.target1[0]], ['z', self.target1[2]]])
                # ['x', 5.5]: 碰1的北面 -> ['x', 5.0]
                self.switchNavigatingState()
                self.next_state_ = self.FlightState.NAVIGATING1
                self.stage = 1
                # 检测1
            elif self.stage == 1:  # 转向
                rospy.loginfo('[[Stage 1 - turn 90 and goto next position]]')
                x = self.detect(self.target1)
                if x == 1:
                    self.commandlist[2] = 'r'
                if x == 2:
                    self.commandlist[2] = 'y'
                if x == 3:
                    self.commandlist[2] = 'b'
                commandstring = 'eeeee'
                commandstring = ''.join(self.commandlist)
                self.publishResult(commandstring)
                self.target_yaw = 0
                self.yaw_x = self.target1[0]
                self.yaw_y = self.target1[1]
                self.adjust_yaw_pos = True
                self.switchNavigatingState()
                self.next_state_ = self.FlightState.NAVIGATING2
                self.stage = 0

        elif self.flight_state_ == self.FlightState.NAVIGATING2:
            if self.stage == 0:  # 导航
                rospy.loginfo('***NAVIGATING2(3->1)...***')
                self.target_yaw = -90
                self.yaw_x = self.target2[0]
                self.yaw_y = self.target2[1]
                self.adjust_yaw_pos = True
                self.navigating_queue_ = deque(
                    [['y', self.target2[1]], ['z', self.target2[2]], ['x', self.target2[0]]])
                self.switchNavigatingState()
                self.stage = 1
            if self.stage == 1:  # 检测
                x = self.detect(self.target2)
                if x == 1:
                    self.commandlist[0] = 'r'
                if x == 2:
                    self.commandlist[0] = 'y'
                if x == 3:
                    self.commandlist[0] = 'b'
                commandstring = ''.join(self.commandlist)
                self.publishResult(commandstring)
                self.next_state_ = self.FlightState.NAVIGATING3

        elif self.flight_state_ == self.FlightState.NAVIGATING3:
            rospy.loginfo('***NAVIGATING3(1->4)...***')
            self.navigating_queue_ = deque(
                [['y', self.target3[1]], ['x', self.target3[0]], ['z', self.target3[2]]])
            self.switchNavigatingState()
            x = self.detect(self.target3)
            if x == 1:
                self.commandlist[3] = 'r'
            if x == 2:
                self.commandlist[3] = 'y'
            if x == 3:
                self.commandlist[3] = 'b'
            commandstring = ''.join(self.commandlist)
            self.publishResult(commandstring)
            self.next_state_ = self.FlightState.NAVIGATING4

        elif self.flight_state_ == self.FlightState.NAVIGATING4:
            rospy.loginfo('***NAVIGATING4(4->5)...***')
            self.target_yaw = -180
            self.yaw_x = self.target3[0]
            self.yaw_y = self.target3[1]
            self.adjust_yaw_pos = True
            self.navigating_queue_ = deque(
                [['y', self.target4[1]], ['x', self.target4[0], ['z', self.target4[2]]]])
            self.switchNavigatingState()
            x = self.detect(self.target4)
            if x == 1:
                self.commandlist[4] = 'r'
            if x == 2:
                self.commandlist[4] = 'y'
            if x == 3:
                self.commandlist[4] = 'b'
            commandstring = ''.join(self.commandlist)
            self.publishResult(commandstring)
            self.next_state_ = self.FlightState.LANDING
        elif self.flight_state_ == self.FlightState.NAVIGATING_FINAL:
            rospy.loginfo('***NAVIGATING5(4->FINAL)...***')
            self.navigating_queue_ = deque(
                [['x', 4], ['y', 12.5], ['x', 7], ['y', 14.5]])
            self.switchNavigatingState()
            self.next_state_ = self.FlightState.LANDING

        elif self.flight_state_ == self.FlightState.LANDING:
            rospy.logwarn('State: LANDING')
            self.publishCommand('land')
        else:
            pass

    # 在向目标点导航过程中，更新导航状态和信息
    def switchNavigatingState(self):
        if len(self.navigating_queue_) == 0:
            self.flight_state_ = self.next_state_
        else:  # 从队列头部取出无人机下一次导航的状态信息
            next_nav = self.navigating_queue_.popleft()
            # TODO 3: 更新导航信息和飞行状态
            self.navigating_dimension_ = next_nav[0]
            self.navigating_destination_ = next_nav[1]
            rospy.loginfo("next_nav:(%s,%.2f)" % (next_nav[0], next_nav[1]))
            self.flight_state_ = self.FlightState.NAVIGATING
            # end of TODO 3

    def detect(self, target_position):
        rospy.logwarn('func: DETECTING_TARGET')
        # 如果无人机飞行高度与标识高度（1.75m）相差太多，则需要进行调整
        if self.t_wu_[2] > target_position[2] + 0.25:
            self.publishCommand('down %d' % int(100 * (self.t_wu_[2] - target_position[2])))
        elif self.t_wu_[2] < target_position[2] - 0.25:
            self.publishCommand('up %d' % int(-100 * (self.t_wu_[2] - target_position[2])))
        # # 如果yaw与90度相差超过正负10度，需要进行旋转调整yaw
        # self.adjust_yaw(self.target_yaw)

        if self.detectTarget() == 1:
            rospy.loginfo('Target detected red.')
            return 1
        elif self.detectTarget() == 2:
            rospy.loginfo('Target detected yellow.')
            return 2
        elif self.detectTarget() == 3:
            rospy.loginfo('Target detected blue.')
            return 3
        return 0



    def adjust_yaw(self, target_yaw):  # 返回值为是否发布命令
        (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
        yaw_diff = yaw - target_yaw if yaw > target_yaw - 180 else yaw + 360 - target_yaw
        # rospy.loginfo('curr_yaw: %d' % yaw)
        # rospy.loginfo('targer_yaw: %d' % target_yaw)
        if yaw_diff > 10:  # clockwise
            self.publishCommand('cw %d' % (int(yaw_diff) if yaw_diff > 15 else 15))
            rospy.loginfo('cw %d' % (int(yaw_diff) if yaw_diff > 15 else 15))
            return True
        elif yaw_diff < -10:  # counterclockwise
            self.publishCommand('ccw %d' % (int(-yaw_diff) if yaw_diff < -15 else 15))
            rospy.loginfo('ccw %d' % (int(-yaw_diff) if yaw_diff < -15 else 15))
            return True
        elif self.adjust_yaw_pos:
            rospy.logwarn("try to adjust yaw position to (x,y)=(%.2f,%.2f)" % (self.yaw_x, self.yaw_y))
            deque.appendleft(self.navigating_queue_, ['x', self.yaw_x])
            deque.appendleft(self.navigating_queue_, ['y', self.yaw_y])
            self.adjust_yaw_pos = False
        return False

    def adjust_pitch(self, target_pitch):  # 返回值为是否发布命令
        (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
        rospy.logwarn("prepare to adjust pitch ...")
        pitch_diff = pitch - target_pitch if pitch > target_pitch - 180 else pitch + 360 - target_pitch
        if pitch_diff > 10:  # clockwise
            self.publishCommand('cw %d' % (int(pitch_diff) if pitch_diff > 15 else 15))
            return True
        elif pitch_diff < -10:  # counterclockwise
            # 发布相应的tello控制命令
            self.publishCommand('ccw %d' % (int(-pitch_diff) if pitch_diff < -15 else 15))
            return True
        return False

    # 判断是否检测到目标
    def detectTarget(self):
        if self.image_ is None:
            return False
        image_copy = self.image_.copy()
        height = image_copy.shape[0]
        width = image_copy.shape[1]

        frame = cv2.resize(image_copy, (width, height), interpolation=cv2.INTER_CUBIC)  # 将图片缩放
        frame = cv2.GaussianBlur(frame, (3, 3), 0)  # 高斯模糊
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
        h, s, v = cv2.split(frame)  # 分离出各个HSV通道
        v = cv2.equalizeHist(v)  # 直方图化
        frame = cv2.merge((h, s, v))  # 合并三个通道

        frame_red = cv2.inRange(frame, self.color_range_red[0], self.color_range_red[1])  # 对原图像和掩模进行位运算
        opened_red = cv2.morphologyEx(frame_red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算
        closed_red = cv2.morphologyEx(opened_red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算
        (image_red, contours_red, hierarchy_red) = cv2.findContours(closed_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓

        # 在contours中找出最大轮廓
        contour_area_max_red = 0
        area_max_contour_red = None
        for c in contours_red:  # 遍历所有轮廓
            contour_area_temp_red = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
            if contour_area_temp_red > contour_area_max_red:
                contour_area_max_red = contour_area_temp_red
                area_max_contour_red = c

        if area_max_contour_red is not None:
            if contour_area_max_red > 50:
                rospy.loginfo("detected: contour_area_max_red = %.2f " % contour_area_max_red)
                return 1

        frame_yellow = cv2.inRange(frame, self.color_range_yellow[0], self.color_range_yellow[1])  # 对原图像和掩模进行位运算
        opened_yellow = cv2.morphologyEx(frame_yellow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算
        closed_yellow = cv2.morphologyEx(opened_yellow, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算
        (image_yellow, contours_yellow, hierarchy_yellow) = cv2.findContours(closed_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓

        # 在contours中找出最大轮廓
        contour_area_max_yellow = 0
        area_max_contour_yellow = None
        for c in contours_yellow:  # 遍历所有轮廓
            contour_area_temp_yellow = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
            if contour_area_temp_yellow > contour_area_max_yellow:
                contour_area_max_yellow = contour_area_temp_yellow
                area_max_contour_yellow = c

        if area_max_contour_yellow is not None:
            if contour_area_max_yellow > 50:
                rospy.loginfo("detected: contour_area_max_yellow = %.2f " % contour_area_max_yellow)
                return 2

        frame_blue = cv2.inRange(frame, self.color_range_blue[0], self.color_range_blue[1])  # 对原图像和掩模进行位运算
        opened_blue = cv2.morphologyEx(frame_blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算
        closed_blue = cv2.morphologyEx(opened_blue, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算
        (image_blue, contours_blue, hierarchy_blue) = cv2.findContours(closed_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓

        # 在contours中找出最大轮廓
        contour_area_max_blue = 0
        area_max_contour_blue = None
        for c in contours_blue:  # 遍历所有轮廓
            contour_area_temp_blue = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
            if contour_area_temp_blue > contour_area_max_blue:
                contour_area_max_blue = contour_area_temp_blue
                area_max_contour_blue = c

        if area_max_contour_blue is not None:
            if contour_area_max_blue > 50:
                rospy.loginfo("detected: contour_area_max_blue = %.2f " % contour_area_max_blue)
                return 3
        return 0


    # 向相关topic发布tello命令
    def publishCommand(self, command_str):
        rospy.logdebug("publish_command:%s" % command_str)
        msg = String()
        msg.data = command_str
        self.commandPub_.publish(msg)
    # 向相关topic发布目标识别、检测结果
    def publishResult(self, command_str):
        msg = String()
        msg.data = command_str
        self.commandResult_.publish(msg)

    # 接收无人机位姿
    def poseCallback(self, msg):
        self.t_wu_ = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.R_wu_ = R.from_quat(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        pass

    # 接收相机图像
    def imageCallback(self, msg):
        try:
            self.image_ = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as err:
            print(err)

    # 接收开始信号
    def startcommandCallback(self, msg):
        self.is_begin_ = msg.data


if __name__ == '__main__':
    cn = ControllerNode()
