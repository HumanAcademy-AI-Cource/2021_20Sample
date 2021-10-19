#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ライブラリのインポート
import boto3
import subprocess
import wave
import os

# コマンドリストの入ったディレクトリの名前
dirname = "./command_lists"

# AWSを使った翻訳の準備
translate = boto3.client(service_name="translate")

# ディレクトリ内のコマンドリストをリストアップ
filenames = []
for name in os.listdir(dirname):
    filenames.append(name)

# リストアップしたコマンドリストのファイル名を表示
print("=======================================")
print("○  翻訳するコマンドリスト")
for name in filenames:
    print("  -  {}".format(name))
print("=======================================")

# 各コマンドリストの中身を読み込む
filedatas = {}
for name in filenames:
    with open(dirname + "/" + name, "r") as f:
        filedatas[name] = f.read()

# 読み込んだコマンドリストの中身を一行ごとにわける
text_datas = []
for name, data in filedatas.items():
    texts = {}
    for index, text in enumerate(data.split("\n")):
        texts[index] = text
    text_datas.append(texts)

# 一行ごとにわけたコマンドリストの中身からコメント部分だけ抜き出す
comment_datas = []
for data in text_datas:
    comments = {}
    for key, text in data.items():
        if text == "" or text[0] != "#":
            continue
        comments[key] = text
    comment_datas.append(comments)

# 抜き出したコメント部分を翻訳
transrate_datas = []
for datas in comment_datas:
    texts = {}
    for key, text in datas.items():
        texts[key] = translate.translate_text(
            Text=text,
            SourceLanguageCode="ja",
            TargetLanguageCode="en"
        )["TranslatedText"].encode("UTF-8")
        print("○  翻訳前: {}".format(text))
        print("○  翻訳後: {}".format(texts[key]))
        print("=================================================")
    transrate_datas.append(texts)

# 翻訳したコマンドリストを保存するディレクトリ名
en_dirname = "./en_command_lists"

# ディレクトリ名作成
if not os.path.isdir(en_dirname):
    os.makedirs(en_dirname)

# 翻訳結果を使って、新しいコマンドリストを作成
for name, text_data, comment_data, transrate_data in zip(filenames, text_datas, comment_datas, transrate_datas):
    with open(en_dirname + "/" + name, "w") as f:
        for key, text in text_data.items():
            data = ""
            if key in comment_data:
                data = transrate_data[key]
            else:
                data = text
            f.write("{}\n".format(data))
