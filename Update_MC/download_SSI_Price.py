from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os
import pandas as pd


def get_data_SSI_price(EXCHANGE, FOLDER_SAVE):
    FOLDER_SAVE = os.path.abspath(FOLDER_SAVE)
    FOLDER_SAVE = FOLDER_SAVE + "/_price_ssi_temp_"
    if not os.path.exists(FOLDER_SAVE):
        os.makedirs(FOLDER_SAVE, exist_ok=True)

    for p_ in os.listdir(FOLDER_SAVE):
        os.remove(FOLDER_SAVE + "/" + p_)

    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", {
        "download.default_directory": FOLDER_SAVE,
        "savefile.default_directory": FOLDER_SAVE
    })

    driver = webdriver.Chrome(options, Service(ChromeDriverManager().install()))
    driver.get("https://iboard.ssi.com.vn/")
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    list_a_tag = soup.find_all("a")
    list_href = [a["href"] for a in list_a_tag]
    list_a_des = [a.text for a in list_a_tag]
    for i in range(len(list_a_tag)):
        if list_a_des[i] == EXCHANGE:
            target_href = list_href[i]
            break
    else:
        driver.quit()
        raise Exception(f'Không có "{EXCHANGE}"')

    driver.get("https://iboard.ssi.com.vn" + target_href)
    time.sleep(3)

    list_dir_old = os.listdir(FOLDER_SAVE)
    button_download = driver.find_element(By.XPATH, "/html/body/div[1]/div[1]/main/div[1]/section[2]/div[2]/div[1]/div[3]/button[1]")
    button_download.click()
    time.sleep(10)

    list_dir_new = os.listdir(FOLDER_SAVE)
    for dir_ in list_dir_new:
        if dir_.endswith(".csv") and dir_ not in list_dir_old:
            filename = dir_
            break

    print("Xem full data tại đây:", f'"{FOLDER_SAVE}/{filename}"')
    driver.quit()

    data = pd.read_csv(f"{FOLDER_SAVE}/{filename}", skiprows=[0])
    return data