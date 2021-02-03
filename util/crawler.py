
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support.expected_conditions import element_to_be_clickable, presence_of_all_elements_located

# https://selenium-python.readthedocs.io/waits.html?highlight=WebDriverWait
driver = webdriver.Edge()

driver.get("https://y.qq.com/portal/singer_list.html")
# driver.get("https://y.qq.com/portal/singer_list.html#page=74&")

# driver.implicitly_wait(10)  # seconds, https://selenium-python.readthedocs.io/waits.html?highlight=WebDriverWait
waiter = WebDriverWait(driver, 10)
with open('singer.txt', 'w', encoding='utf-8') as file:
    i = 1
    while True:
        print('page', i)
        i += 1
        try:
            for e in waiter.until(presence_of_all_elements_located((By.CLASS_NAME, "singer_list_txt__item"))):
                print(e.text, file=file)
                print(e.text)
        except Exception as e:
            print(e)
            driver.refresh()
            i -= 1
            continue

        try:
            button = waiter.until(element_to_be_clickable((By.CLASS_NAME, "next js_pageindex")))
            button.click()
            driver.refresh()  # https://selenium-python.readthedocs.io/api.html?highlight=refresh
        except Exception as e:
            print(e)
            break

driver.close()
