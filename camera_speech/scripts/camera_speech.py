#!/usr/bin/env python
# -*- coding: utf-8 -*-


# 必要なライブラリをインポート
import rospy
import cv2
import subprocess
import roslib.packages
import boto3
import wave
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraSpeech(object):
    def __init__(self):
        rospy.Subscriber("/image_raw", Image, self.imageCB)
        self.image = None
        self.enable_process = False

    def process(self):
        # -------------------------------
        # カメラ画像を保存する
        # -------------------------------
        pkg_path = roslib.packages.get_pkg_dir("camera_speech")
        target_file = pkg_path + "/scripts/camera.jpg"
        cv2.imwrite(target_file, self.image)
        image = cv2.imread(target_file)

        # -------------------------------
        # 画像認識して表示用の画像を作成
        # -------------------------------
        detect_data = self.detectLabels(target_file)
        for d in detect_data:
            for b in d["Instances"]:
                x = int(b["BoundingBox"]["Left"] * image.shape[1])
                y = int(b["BoundingBox"]["Top"] * image.shape[0])
                w = int(b["BoundingBox"]["Width"] * image.shape[1])
                h = int(b["BoundingBox"]["Height"] * image.shape[0])
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 241, 255), 2)
                text_size = cv2.getTextSize(d["Name"], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                image = cv2.rectangle(image, (x, y - text_size[0][1] - 10), (x + text_size[0][0], y), (0, 241, 255), -1)
                image = cv2.putText(image, d["Name"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)

        # -------------------------------
        # 画像認識の結果を翻訳する
        # -------------------------------
        trans_text = ""
        for i in range(len(detect_data)):
            trans_text += detect_data[i]["Name"] + "\n"
        transrate_data = self.transrate(trans_text)
        sp_transrate_data = transrate_data.split("\n")[:-1]
        
        for i in range(len(detect_data)):
            original_name = detect_data[i]["Name"]
            trans_name = sp_transrate_data[i]
            confidence = detect_data[i]["Confidence"]
            print(trans_name + "({})({:.2f}%)".format(original_name, confidence))
            if i == 4:
                break
        print("=================================================")

        # -------------------------------
        # 翻訳結果の文章化
        # -------------------------------
        text = ""
        for i in range(len(sp_transrate_data)):
            text += sp_transrate_data[i]
            if i == 2:
                break
            else:
                text += ","
        text += "が見つかりました。".decode('utf-8')

        # -------------------------------
        # 翻訳結果を音声合成する
        # -------------------------------
        speech_data = self.synthesizeSpeech(text)
        filename = pkg_path + "/scripts/speech.wav"
        wave_data = wave.open(filename, 'wb')
        wave_data.setnchannels(1)
        wave_data.setsampwidth(2)
        wave_data.setframerate(16000)
        wave_data.writeframes(speech_data.read())
        wave_data.close()

        cv2.imshow('Result', cv2.resize(image, dsize=None, fx=0.5, fy=0.5))
        key = cv2.waitKey(1)

        # 保存したWAVデータを再生
        subprocess.check_call('aplay -D plughw:0 {}'.format(filename), shell=True)

        print("=================================================")
        print("[s]キーを押すと画像認識・翻訳・音声合成スタート")
        print("=================================================")

    def detectLabels(self, path):
        # AWSで画像認識
        rekognition = boto3.client("rekognition")
        with open(path, 'rb') as f:
            return rekognition.detect_labels(
                Image={'Bytes': f.read()},
            )["Labels"]
        return []

    def transrate(self, text):
        # AWSで翻訳
        translate = boto3.client(service_name="translate")
        return translate.translate_text(
            Text=text,
            SourceLanguageCode="en",
            TargetLanguageCode="ja"
        )["TranslatedText"]

    def synthesizeSpeech(self, text):
        # AWSで音声合成
        polly = boto3.client(service_name="polly")
        return polly.synthesize_speech(
            Text=text,
            OutputFormat='pcm',
            VoiceId='Mizuki'
        )['AudioStream']

    def imageCB(self, msg):
        # カメラ画像を受け取る
        self.image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow('Camera', cv2.resize(self.image, dsize=None, fx=0.5, fy=0.5))
        key = cv2.waitKey(1)
        if key == ord('s'):
            self.enable_process = True

    def run(self):
        print("=================================================")
        print("[s]キーを押すと画像認識・翻訳・音声合成スタート")
        print("=================================================")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.enable_process:
                self.enable_process = False
                self.process()
            rate.sleep()


if __name__ == '__main__':
    # ノードを宣言
    rospy.init_node('camera_speech_node')
    CameraSpeech().run()
