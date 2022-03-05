#! /Users/glebsokolov/opt/anaconda3/bin/python

import requests
import wget
from bs4 import BeautifulSoup
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

url = 'http://library.lol/main/6393A6CD6994DA538EADEBFE61C1F907'

bs = BeautifulSoup(requests.Session().get(url).text, 'html.parser')
link = bs.find_all(lambda tag: tag.name == 'a' and 'Cloudflare' in tag.text)[0]['href']
ext = '.pdf' if 'pdf' in link else '.epub'
name = bs.find('h1').get_text()
wget.download(link, out='/Users/glebsokolov/Downloads/' + name + ext)
#%%
