from time import sleep
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import os

driver = webdriver.Chrome()
to_location = 'BLR'
url = 'https://www.kayak.co.in/flights/IXC-{to_location}/2024-11-08/2024-11-15?sort=bestflight_a'.format(to_location=to_location)

driver.get(url)
