from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, \
    QLabel, QLineEdit, QComboBox, QSpinBox, QTextEdit, \
    QProgressBar, QToolButton
from PyQt6.QtCore import QRunnable, pyqtSignal, QObject, QThreadPool
from PyQt6.QtGui import QIcon
from PyQt6 import uic
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2
import os
import numpy as np
from roboflow import Roboflow
import json
import math
import threading


class WorkerSignals(QObject):
    frame_num = pyqtSignal(int)
    num = pyqtSignal(int)

    x_center = pyqtSignal(int)
    y_center = pyqtSignal(int)
    width = pyqtSignal(int)
    height = pyqtSignal(int)
    max_confidence = pyqtSignal(float)

    result = pyqtSignal(str)


class PredictionThread(QRunnable):
    def __init__(self, video, model, subfilename, frames_file, numofframes, stop_predict):
        super().__init__()
        self.signals = WorkerSignals()
        self.video = video
        self.model = model
        self.subfilename = subfilename
        self.frames_file = frames_file
        self.numofframes = numofframes
        self.stop_prediction = stop_predict

    def run(self):
        frame_num = 1
        num = 1
        while self.video.isOpened():
            if self.stop_prediction.is_set():  # 检查是否应该停止预测
                break
            # 读取每一帧
            ret, frame = self.video.read()
            if ret:

                data = self.model.predict(frame, confidence=40, overlap=30).json()
                if not data['predictions']:
                    pass
                else:
                    max_confidence = 0
                    for instance in data['predictions']:
                        if instance['confidence'] > max_confidence:
                            label = instance['class']
                            max_confidence = instance['confidence']
                            x_center = int(instance['x'])
                            y_center = int(instance['y'])
                            width = int(instance['width'])
                            height = int(instance['height'])
                            x1 = int(instance['x'] - instance['width'] * 0.5)
                            x2 = int(instance['x'] + instance['width'] * 0.5)
                            y1 = int(instance['y'] - instance['height'] * 0.5)
                            y2 = int(instance['y'] + instance['height'] * 0.5)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(frame, str(max_confidence), (x1, y2 + 23),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    self.signals.frame_num.emit(frame_num)
                    self.signals.x_center.emit(x_center)
                    self.signals.y_center.emit(y_center)
                    self.signals.width.emit(width)
                    self.signals.height.emit(height)
                    self.signals.max_confidence.emit(max_confidence)

                self.signals.num.emit(num)
                # print(str(frame_num) + '/' + str(self.numofframes))
                file_name = self.subfilename + '_' + str(frame_num) + '.jpg'
                frame_num += 1
                num += 1
                path_name = os.path.join(self.frames_file, file_name)

                cv2.imwrite(path_name, frame)

            else:
                break
        self.signals.result.emit("Prediction finished")


class UI(QWidget):
    def __init__(self):
        super().__init__()

        self.index_end = None
        self.index_start = None
        self.confidence = None
        self.frames = None
        self.x_center = None
        self.y_center = None
        self.width = None
        self.height = None
        self.model = None
        self.save_path = None
        self.numofframes = None
        self.video = None
        self.subfilename = None
        self.target = None
        # self.stop_prediction = False  # 用于跟踪是否停止预测
        self.stop_prediction = threading.Event()
        self.prediction_thread = None

        self.setStyleSheet(
            "background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #fdfbfb, stop: 1 #ebedee);")

        button_style1 = '''
        QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #008080, stop:1 #4B0082);
    border: 2px solid #333;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #4B0082, stop:1 #008080);
    border: 2px solid #555;
}
'''

        button_style2 = '''
        QPushButton {
    background-color: rgba(0, 0, 255, 0.5);
    border: none;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    backdrop-filter: blur(10px);
}
QPushButton:hover {
    background-color: rgba(0, 0, 255, 0.8);
}
'''

        button_style3 = '''
        QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #3498db, stop:1 #2980b9);
    border: none;
    color: white;
    padding: 10px 10px;
    border-radius: 10px;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #2980b9, stop:1 #3498db);
}
        '''

        button_style4 = '''
        QPushButton {
    background-color: #96e6a1;
    border: none;
    color: white;
    padding: 10px 20px;
    border-radius: 20px;
    font-size: 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #CD853F;
}
        '''

        toolbtn_style = '''
       QToolButton {
    border: 1px solid #3498db;
    border-radius: 1px;
    background-color: transparent;
    padding: 1px;
}
QToolButton:hover {
    background-color: #3498db;
    color: white;
}
        '''

        progressbar_style1 = '''
        QProgressBar {
    border: 2px solid #3498db;
    border-radius: 10px;
    background-color: white;
    height: 10px;
}
QProgressBar::chunk {
    background-color: #3498db;
    border-radius: 10px;
}

        '''

        progressbar_style2 = '''
        QProgressBar {
    border: 2px solid #ccc;
    border-radius: 5px;
    background-color: #f0f0f0;
    height: 10px;
}
QProgressBar::chunk {
    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
        stop:0 #3498db, stop:1 #2980b9);
    border-radius: 5px;
}
        '''

        uic.loadUi("v3.ui", self)

        # set logo
        self.setWindowIcon(QIcon("logo/ai.png"))

        # terminal
        self.textInfo = self.findChild(QTextEdit, "textEdit_info")

        # load video
        self.loadVideo_button = self.findChild(QPushButton, "pushButton_load")
        self.loadVideo_button.clicked.connect(self.load_video)
        self.loadVideo_button.setStyleSheet(button_style1)

        # save path
        self.ProcessingVideo_savepath = self.findChild(QLineEdit, "lineEdit_savepath")
        self.ProcessingVideo_selectpath = self.findChild(QToolButton, "toolButton_selectpath")
        self.ProcessingVideo_selectpath.setStyleSheet(toolbtn_style)
        self.ProcessingVideo_selectpath.clicked.connect(self.get_save_path)

        # get frames
        self.ProcessingVideo_getframes = self.findChild(QPushButton, "pushButton_getframes")
        self.ProcessingVideo_getframes.clicked.connect(self.get_frames)
        self.ProcessingVideo_getframes.setStyleSheet(button_style2)
        self.Processing_getframes = self.findChild(QProgressBar, "progressBar_getframes")
        self.Processing_getframes.setStyleSheet(progressbar_style1)
        self.Processing_getframes.setValue(0)

        # Load account
        self.loadAccount_token = self.findChild(QLineEdit, "lineEdit_token")
        self.loadAccount_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.loadAccount_connect = self.findChild(QPushButton, "pushButton_connect")
        self.loadAccount_connect.clicked.connect(self.account_connect)
        self.loadAccount_connect.setStyleSheet(button_style4)
        self.loadAccount_project = self.findChild(QComboBox, "comboBox_project")

        # choose model
        self.choose_model_version = self.findChild(QSpinBox, "spinBox_version")
        self.choose_model_confirm = self.findChild(QPushButton, "pushButton_confirm")
        self.choose_model_confirm.clicked.connect(self.choose_model)
        self.choose_model_confirm.setStyleSheet(button_style4)

        # start prediction
        self.start_prediction_start = self.findChild(QPushButton, "pushButton_start")
        self.start_prediction_start.clicked.connect(self.start_prediction)
        self.start_prediction_start.setStyleSheet(button_style1)
        self.start_prediction_finish = self.findChild(QLabel, "label_finish")
        self.start_prediction_progress = self.findChild(QProgressBar, "progressBar_progress")
        self.start_prediction_progress.setStyleSheet(progressbar_style2)
        self.start_prediction_progress.setValue(0)

        # stop prediction
        self.pushbutton_stop = self.findChild(QPushButton, "pushButton_stop")
        self.pushbutton_stop.clicked.connect(self.stop_prediction_button_clicked)
        self.pushbutton_stop.setStyleSheet(button_style4)

        # write2video
        self.ProcessingVideo_write2video = self.findChild(QPushButton, "pushButton_write2video")
        self.ProcessingVideo_write2video.clicked.connect(self.write2video)
        self.ProcessingVideo_write2video.setStyleSheet(button_style2)

        # show video
        self.ProcessingVideo_showvideo = self.findChild(QPushButton, "pushButton_showvideo")
        self.ProcessingVideo_showvideo.clicked.connect(self.show_video)
        self.ProcessingVideo_showvideo.setStyleSheet(button_style2)

        # add plt
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.horizontalLayout.addWidget(self.canvas)

        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_hover)

        # update
        self.button_update = self.findChild(QPushButton, "pushButton_update")
        self.button_update.clicked.connect(self.plot_tlp)
        self.button_update.setStyleSheet(button_style3)
        self.plot_info = self.findChild(QLabel, "label_plotinfo")

        # center
        self.button_center = self.findChild(QPushButton, "pushButton_center")
        self.button_center.clicked.connect(self.plot_center)
        self.button_center.setStyleSheet(button_style3)

        # confidence
        self.button_confidence = self.findChild(QPushButton, "pushButton_confidence")
        self.button_confidence.clicked.connect(self.plot_confidence)
        self.button_confidence.setStyleSheet(button_style3)

        # update coordinates
        self.combox_startframe = self.findChild(QComboBox, "comboBox_startframe")
        self.combox_endframe = self.findChild(QComboBox, "comboBox_endframe")
        self.update_confirm = self.findChild(QPushButton, "pushButton_confirm_2")
        self.update_confirm.clicked.connect(self.update_frame)
        self.update_confirm.setStyleSheet(button_style4)

        # Velocity Calculator
        self.lineedit_scale = self.findChild(QLineEdit, "lineEdit_scale")
        self.lineedit_fps = self.findChild(QLineEdit, "lineEdit_fps")
        self.lineedit_angle = self.findChild(QLineEdit, "lineEdit_angle")
        self.button_velocity = self.findChild(QPushButton, "pushButton_velocity")
        self.button_velocity.clicked.connect(self.velocity_caclulation)
        self.button_velocity.setStyleSheet(button_style1)
        self.velocity_info = self.findChild(QTextEdit, "textEdit_velocity")
        self.button_clear = self.findChild(QPushButton, "pushButton_clear")
        self.button_clear.clicked.connect(self.text_clear)
        self.button_clear.setStyleSheet(button_style1)

        # multi thread
        self.threadpool = QThreadPool()

    def load_video(self):
        self.video = None
        self.target = 0
        filename, pathname, self.target = select_video()
        if self.target == 1:
            self.subfilename, self.video, self.numofframes = get_video_info(filename)
            self.index_start = 0
            self.index_end = self.numofframes
            self.textInfo.append("Load successfully")
            self.textInfo.append("User selected: {}".format(pathname + filename))
            self.textInfo.append("Num of frames: {}".format(self.numofframes))
        else:
            self.textInfo.append("Fail to load video, User selected Cancel")

    def get_save_path(self):
        root = tk.Tk()
        root.withdraw()
        try:
            self.save_path = filedialog.askdirectory()
            if self.save_path:
                self.ProcessingVideo_savepath.setText(self.save_path)
            else:
                self.textInfo.append("Folder selection canceled.")
                self.save_path = None
        except:
            self.textInfo.append("Folder selection canceled!")

    def get_frames(self):
        if self.target != 1:
            self.textInfo.append("Please load video")
        else:
            if self.save_path is None:
                self.textInfo.append("Please select the save path")
            else:
                video_file = os.path.join(self.save_path, 'VideoFrames')
                frames_file = os.path.join(video_file, self.subfilename)
                if not os.path.exists(video_file):
                    os.mkdir(video_file)
                if not os.path.exists(frames_file):
                    os.mkdir(frames_file)

                frame_num = 0
                self.Processing_getframes.setValue(0)
                self.Processing_getframes.setRange(0, self.numofframes)
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while self.video.isOpened():
                    ret, frame = self.video.read()  # 读取一帧

                    if ret:
                        frame_num += 1
                        self.Processing_getframes.setValue(frame_num)
                        file_name = self.subfilename + '_' + str(frame_num) + '.jpg'
                        path_name = os.path.join(frames_file, file_name)

                        cv2.imwrite(path_name, frame)  # 保存一帧
                    else:
                        break
                self.textInfo.append("Successfully completed")
                # self.video.release()
                # cv2.destroyAllWindows()

    def account_connect(self):
        token = self.loadAccount_token.text()
        try:
            rf = Roboflow(api_key=token)
            data = rf.workspace()
            data = json.loads(str(data))
            projects = data["projects"]
            for project in projects:
                project_name = project.split('/')[-1]
                self.loadAccount_project.addItem(project_name)
            self.textInfo.append("connection succeeded")
        except:
            self.textInfo.append("token wrong")
            self.loadAccount_project.clear()

    def choose_model(self):
        self.model = None
        try:
            project = self.loadAccount_project.currentText()
            version = self.choose_model_version.value()
            token = self.loadAccount_token.text()
            self.model = open_model(token, project, version)
            self.textInfo.append("Model loaded successfully")
        except:
            self.textInfo.append("Model loading failed")

    def start_prediction(self):
        if self.target != 1:
            self.textInfo.append("Please load video")
        else:
            if self.save_path is None:
                self.textInfo.append("Please select the save path")
            else:
                if self.model is None:
                    self.textInfo.append("Please load model")
                else:
                    video_file = os.path.join(self.save_path, 'DetectFrames')
                    frames_file = os.path.join(video_file, self.subfilename)
                    if not os.path.exists(video_file):
                        os.mkdir(video_file)
                    if not os.path.exists(frames_file):
                        os.mkdir(frames_file)

                    # 如果当前存在运行中的预测线程，则先停止它
                    # if self.prediction_thread and self.prediction_thread.is_alive():
                    #     self.stop_prediction.set()  # 停止预测线程

                    # 启动新的预测线程
                    if self.stop_prediction.is_set():
                        self.stop_prediction.clear()  # 清除停止标志

                    self.x_center = []
                    self.y_center = []
                    self.width = []
                    self.height = []
                    self.frames = []
                    self.confidence = []
                    self.start_prediction_progress.setValue(0)
                    self.start_prediction_progress.setRange(0, self.numofframes)
                    self.combox_startframe.clear()
                    self.combox_endframe.clear()
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.start_prediction_finish.setText("reasoning...")

                    # Start the worker in a thread
                    self.prediction_thread = PredictionThread(self.video, self.model, self.subfilename, frames_file,
                                                              self.numofframes, self.stop_prediction)

                    self.prediction_thread.signals.frame_num.connect(self.update_info)
                    self.prediction_thread.signals.num.connect(self.update_progass)
                    self.prediction_thread.signals.x_center.connect(self.update_x_center)
                    self.prediction_thread.signals.y_center.connect(self.update_y_center)
                    self.prediction_thread.signals.width.connect(self.update_w)
                    self.prediction_thread.signals.height.connect(self.update_h)
                    self.prediction_thread.signals.max_confidence.connect(self.update_confidence)
                    self.prediction_thread.signals.result.connect(self.update_finish_label)

                    self.threadpool.start(self.prediction_thread)

    def stop_prediction_button_clicked(self):
        # "Stop" 按钮点击事件处理程序
        self.stop_prediction.set()
        self.textInfo.append("Stopping prediction...")

    def update_info(self, frame_num):
        self.combox_startframe.addItem(str(frame_num))
        self.combox_endframe.addItem(str(frame_num))
        self.frames.append(frame_num)

    def update_progass(self, num):
        self.start_prediction_progress.setValue(num)

    def update_x_center(self, x_center):
        self.x_center.append(x_center)

    def update_y_center(self, y_center):
        self.y_center.append(y_center)

    def update_w(self, width):
        self.width.append(width)

    def update_h(self, height):
        self.height.append(height)

    def update_confidence(self, max_confidence):
        self.confidence.append(max_confidence)

    def update_finish_label(self, result):
        self.start_prediction_finish.setText(result)

    def write2video(self):
        if self.save_path is None:
            self.textInfo.append("Please select the save path")
        else:

            root = tk.Tk()
            root.withdraw()  # 隐藏根窗口

            try:
                folder_path = filedialog.askdirectory()  # 选择文件夹
                if folder_path:

                    # 获取文件列表
                    files = os.listdir(folder_path)
                    filename = folder_path.split('/')[-1] + ".avi"

                    video_file = os.path.join(self.save_path, 'DetectVideo')
                    if not os.path.exists(video_file):
                        os.mkdir(video_file)

                    subfolder = os.path.join(video_file, folder_path.split('/')[-1])
                    if not os.path.exists(subfolder):
                        os.mkdir(subfolder)

                    pathname = os.path.join(subfolder, filename)
                    # 设置视频参数
                    fps = 24
                    frame_size = (1280, 1024)
                    video = cv2.VideoWriter(pathname, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, frame_size)

                    images = [os.path.join(folder_path, file) for file in files]
                    img_paths = sorted(images, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
                    # 遍历所有图片
                    for path in img_paths:
                        # 读入图片
                        img = cv2.imread(path)

                        # 调整尺寸
                        img = cv2.resize(img, frame_size)
                        # print(path)
                        # 写入视频
                        video.write(img)

                    # 释放视频对象
                    video.release()
                    self.textInfo.append("Save to" + subfolder)
                    self.textInfo.append("Write finished.")
                else:
                    self.textInfo.append("Folder selection canceled.")
            except:
                self.textInfo.append("Folder selection canceled.")

    def show_video(self):
        root = tk.Tk()
        root.withdraw()

        try:
            folder_path = filedialog.askdirectory()
            if folder_path:
                self.textInfo.append("Selected folder:" + folder_path)

                files = os.listdir(folder_path)

                images = [os.path.join(folder_path, file) for file in files]
                img_paths = sorted(images, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))

                for path in img_paths:
                    # 读入图片
                    img = cv2.imread(path)

                    cv2.imshow('frame', img)
                    cv2.waitKey(100)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                cv2.destroyAllWindows()
            else:
                self.textInfo.append("Folder selection canceled.")
        except tk.TclError:
            self.textInfo.append("Folder selection canceled.")

    def plot_tlp(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        try:
            x1_coords = [int(a - b/2) for a, b in zip(self.x_center, self.width)]
            y1_coords = [int(a - b/2) for a, b in zip(self.y_center, self.height)]
            x = x1_coords[self.index_start:self.index_end + 1]
            y = y1_coords[self.index_start:self.index_end + 1]
            frame = self.frames[self.index_start:self.index_end + 1]

            ax.scatter(frame, x, label='X Coordinate', marker='o')
            ax.plot(frame, y, label='Y Coordinate', marker='x')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Coordinates')
            ax.set_title('Top Left Point Coordinates vs Frames')
            ax.legend()

            self.canvas.draw()
        except:
            self.textInfo.append("Please enter parameters.")

    def plot_center(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        try:
            x = self.x_center[self.index_start:self.index_end + 1]
            y = self.y_center[self.index_start:self.index_end + 1]
            frame = self.frames[self.index_start:self.index_end + 1]

            ax.scatter(frame, x, label='X Coordinate', marker='o')
            ax.plot(frame, y, label='Y Coordinate', marker='x')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Coordinates')
            ax.set_title('Center Point Coordinates vs Frames')
            ax.legend()

            self.canvas.draw()
        except:
            self.textInfo.append("Please enter parameters.")

    def plot_confidence(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        try:
            confidence = self.confidence[self.index_start:self.index_end + 1]
            frame = self.frames[self.index_start:self.index_end + 1]

            ax.plot(frame, confidence, label='confidence')

            ax.set_xlabel('Frame')
            ax.set_ylabel('Confidence')
            ax.set_title('Confidence Plot')
            ax.legend()

            self.canvas.draw()
        except:
            self.textInfo.append("Please enter parameters.")

    def on_mouse_hover(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.plot_info.setText(f'X: {x:.2f}, Y: {y:.2f}')
        else:
            self.plot_info.setText("")

    def update_frame(self):
        start_frame = int(self.combox_startframe.currentText())
        end_frame = int(self.combox_endframe.currentText())
        frames_array = np.array(self.frames)

        self.index_start = np.where(frames_array == start_frame)[0][0]
        self.index_end = np.where(frames_array == end_frame)[0][0]

    def velocity_caclulation(self):
        try:
            scale = float(self.lineedit_scale.text())
            fps = int(self.lineedit_fps.text())
            angle = float(self.lineedit_angle.text())

            radians = math.radians(angle)
            tangent_value = math.tan(radians)

            start_frame = int(self.combox_startframe.currentText())
            end_frame = int(self.combox_endframe.currentText())
            numf = end_frame - start_frame
            t = numf / fps

            x_center = self.x_center[self.index_start:self.index_end + 1]
            nump_x_center = x_center[0] - x_center[-1]
            y_center = self.y_center[self.index_start:self.index_end + 1]
            nump_y_center = y_center[0] - y_center[-1]

            dist_x_center = scale * nump_x_center
            dist_y_center = scale * nump_y_center

            velocity_x_center = dist_x_center / t
            velocity_y_center = dist_y_center / t
            velocity_z_center = tangent_value * velocity_x_center
            self.velocity_info.append("Velocity X_Center is {}".format(velocity_x_center) + " m/s")
            self.velocity_info.append("Velocity Y_Center is {}".format(velocity_y_center) + " m/s")
            self.velocity_info.append("Velocity Z_Center is {}".format(velocity_z_center) + " m/s")
            self.velocity_info.append("--------------------------------------------------------")

            x1_coords = [int(a - b / 2) for a, b in zip(self.x_center, self.width)]
            y1_coords = [int(a - b / 2) for a, b in zip(self.y_center, self.height)]
            x_tl = x1_coords[self.index_start:self.index_end + 1]
            nump_x_tl = x_tl[0] - x_tl[-1]
            y_tl = y1_coords[self.index_start:self.index_end + 1]
            nump_y_tl = y_tl[0] - y_tl[-1]

            dist_x_tl = scale * nump_x_tl
            dist_y_tl = scale * nump_y_tl

            velocity_x_tl = dist_x_tl / t
            velocity_y_tl = dist_y_tl / t
            velocity_z_tl = tangent_value * velocity_x_tl
            self.velocity_info.append("Velocity X_tl is {}".format(velocity_x_tl) + " m/s")
            self.velocity_info.append("Velocity Y_tl is {}".format(velocity_y_tl) + " m/s")
            self.velocity_info.append("Velocity Z_tl is {}".format(velocity_z_tl) + " m/s")

            self.velocity_info.append("======================================")

        except:
            self.velocity_info.append("Please enter parameters.")

    def text_clear(self):
        self.velocity_info.clear()



def open_model(token, project, version):
    rf = Roboflow(api_key=token)
    project = rf.workspace().project(project)
    model = project.version(version).model
    return model


def select_video():
    filename = ''
    pathname = ''

    root = tk.Tk()
    root.withdraw()  # 隐藏根窗口

    filename_temp = os.path.basename(os.path.normpath(os.path.expanduser(os.path.normpath('~'))))
    types = [('Video files', '*.avi *.mp4'), ('All files', '*.*')]
    file_path_name = os.path.join(filename_temp, filename)
    filename = filedialog.askopenfilename(filetypes=types, initialdir=os.path.dirname(file_path_name))
    target = 0

    if filename:
        pathname = os.path.dirname(filename)
        target = 1

    return filename, pathname, target


def get_video_info(filename):
    subfilename = os.path.splitext(os.path.basename(filename))[0]
    video = cv2.VideoCapture(filename)
    num_of_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return subfilename, video, num_of_frame


def main():
    app = QApplication([])
    window = UI()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
