#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ライブラリのインポート
import boto3
import subprocess
import wave


# 翻訳したい文章(英語)を入力する
print("日本語を入力すると、英語に翻訳して音声合成します。")
input_text = raw_input("入力: > ")

# AWSを使う準備
translate = boto3.client(service_name="translate")
polly = boto3.client(service_name="polly")

# 文章を翻訳
translate_text = translate.translate_text(
    Text=input_text,
    SourceLanguageCode="ja",
    TargetLanguageCode="en"
)["TranslatedText"].encode("UTF-8")

# 結果を表示
print("=================================================")
print("○  翻訳前: {}".format(input_text))
print("=================================================")
print("○  翻訳後: {}".format(translate_text))
print("=================================================")

# 音声合成を実行
speech_data = polly.synthesize_speech(
    Text=translate_text,
    OutputFormat='pcm',
    VoiceId='Salli'
)['AudioStream']

# 音声合成の結果をWAVデータとして保存する
wave_data = wave.open("speech.wav", 'wb')
wave_data.setnchannels(1)
wave_data.setsampwidth(2)
wave_data.setframerate(16000)
wave_data.writeframes(speech_data.read())
# ファイルを閉じる
wave_data.close()

# 保存したWAVデータを再生
subprocess.check_call('aplay -D plughw:0 {}'.format("speech.wav"), shell=True)


