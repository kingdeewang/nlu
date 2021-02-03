
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
import time

# https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/#downloads
# https://selenium-python.readthedocs.io/waits.html?highlight=WebDriverWait
# copy utility.modelsDirectory + 'MicrosoftWebDriver.exe' to C:\Users\asus\AppData\Local\Programs\Python\Python36! not C:\Users\asus\AppData\Local\Programs\Python\Python36/scripts
driver = webdriver.Edge()

driver.get("https://192.168.1.1/auth.html")

# print(driver.page_source)

# driver.implicitly_wait(10)  # seconds, https://selenium-python.readthedocs.io/waits.html?highlight=WebDriverWait
waiter = WebDriverWait(driver, 10)

moreInformationDropdownLink = driver.find_element_by_id("moreInformationDropdownLink")
moreInformationDropdownLink.click()

invalidcert_continue = driver.find_element_by_id("invalidcert_continue")
invalidcert_continue.click()

# https://www.cnblogs.com/ericazy/p/6894060.html
# https://www.cnblogs.com/liyuhang/p/6661835.html
time.sleep(3)
driver.switch_to.frame("authFrm")

print(driver.page_source)

userName = driver.find_element_by_name("userName")
pwd = driver.find_element_by_name("pwd")

# from selenium.webdriver.common.action_chains import ActionChains
#
# action = ActionChains(driver)
# time.sleep(5)
# action.move_to_element(userName)
# time.sleep(5)
# action.click(userName)
# time.sleep(5)
# action.send_keys(Keys.ARROW_DOWN)
# time.sleep(5)
# action.send_keys(Keys.ENTER)
# time.sleep(5)
# action.send_keys(Keys.ENTER)
# time.sleep(5)
# action.perform()
# time.sleep(5)
# userName.click()

userName.send_keys('zhoulizhi')
# time.sleep(5)
# userName.send_keys(Keys.ENTER)

# userName.send_keys(Keys.ENTER)

# userName.send_keys('',)

#
#
# pwd.send_keys('qaz!@#123')
# time.sleep(5)
# driver.find_element_by_name("Submit").click()

# <input name="Submit" class="button" type="submit" value="Login">

# driver.close()
