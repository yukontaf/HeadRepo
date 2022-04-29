#! /Users/glebsokolov/opt/anaconda3/bin/python

import os, sys, pytesseract, glob
from PIL import Image
#name=os.environ["KMVAR_name"]

search_dir = "/Users/glebsokolov/Documents/"
files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
files = [f for f in files if f.endswith("png")]
files.sort(key=lambda x: os.path.getmtime(x))



name = pytesseract.image_to_string(Image.open(files[-1]), lang="eng+rus")

fname = "/Users/glebsokolov/Library/Containers/com.QReader.MarginStudyMac-setapp/Data/Documents/DB-K1NDFR/Python Luchshie praktiki i instrumenty 3 izd  2021 Mikhal Yavorski Tarek Ziade.pdf"
name = " ".join(name.split()).lower().title()
root = os.path.dirname(fname)
ext = os.path.splitext(fname)[1]
name += ext
print(name)
