# -*- coding: utf-8 -*-
from python.test_download_data import download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--url', help='url')
parser.add_argument('--filename', help='filename')

args = parser.parse_args()

download(args.url, args.filename)
