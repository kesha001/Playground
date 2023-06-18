import os, sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


def demo_form_submit(driver):
    '''
    Fills form on demo website form and submits it
    '''

    driver.get("https://testpages.herokuapp.com/styled/basic-html-form-test.html")

    # input username
    username_inp = driver.find_element(By.XPATH, "//form//input[@name='username']")
    username_inp.send_keys("Dan")

    # input password
    pass_inp = driver.find_element(By.XPATH, "//form//input[@name='password']")
    pass_inp.send_keys("pass")

    # input in text area
    # comment_inp = driver.find_element(By.XPATH, "//form//textarea[@name='comments']")
    comment_inp = driver.find_element(By.CSS_SELECTOR, "textarea[name='comments']")
    comment_inp.send_keys("Here is my text input by css selector")

    # uploading file
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(os.path.sep, ROOT_DIR,'image.png')
    print(image_path)
    file_inp = driver.find_element(By.XPATH, "//form//input[@name='filename']")
    # file_inp.send_keys("D:/playground/Python/Selenium_Course/demo_examples/image.png")
    file_inp.send_keys(image_path)

    # checkboxes
    checkbox_inp1 = driver.find_element(By.XPATH, "//form//input[@value='cb1']")
    checkbox_inp1.click()
    checkbox_inp3 = driver.find_element(By.XPATH, "//form//input[@value='cb3']")
    checkbox_inp3.click()

    # multiple selection
    myElemA = driver.find_element(By.XPATH, "//form//select[@name='multipleselect[]']/option[@value='ms1']")
    myElemB = driver.find_element(By.XPATH, "//form//select[@name='multipleselect[]']/option[@value='ms2']")
    myElemC = driver.find_element(By.XPATH, "//form//select[@name='multipleselect[]']/option[@value='ms4']")
    ActionChains(driver).key_down(Keys.CONTROL).click(myElemA).key_up(Keys.CONTROL).perform()
    ActionChains(driver).key_down(Keys.CONTROL).click(myElemB).key_up(Keys.CONTROL).perform()
    ActionChains(driver).key_down(Keys.CONTROL).click(myElemC).key_up(Keys.CONTROL).perform()

    # dropdown button
    select_inp = driver.find_element(By.XPATH, "//form//select/option[@value='dd2']")
    select_inp.click()

    # submit form
    submit_form = driver.find_element(By.XPATH, "//form//input[@value='submit']")
    submit_form.click()


def progress_bar_test(driver):
    '''
    Press the download button and wait untill it finishes
    '''
    driver.get("https://jqueryui.com/resources/demos/progressbar/download.html")

    download_button = driver.find_element(By.ID, 'downloadButton')
    download_button.click()

    # wait until progress bar will be finished
    download_completed = WebDriverWait(driver=driver, timeout=20).until(
        EC.text_to_be_present_in_element(
            (By.CLASS_NAME, 'progress-label'), # Filter element
            'Complete!'
        )
    )

    if download_completed:
        progress_element_status = driver.find_element(By.CLASS_NAME, 'progress-label')
        print(progress_element_status.text)



if __name__ == "__main__":

    option = webdriver.ChromeOptions()

    # Path for chromedriver.exe to find browser
    driver_path="D:/SeleniumDrivers/Chrome"
    os.environ['PATH'] += driver_path
    # Path for chromedriver.exe to find browser. For default chrome browser location it is not needed
    option.binary_location = 'C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe'
    
    
    # uncomment to keep browser open after finishing script
    # option.add_experimental_option("detach", True)
    option.add_experimental_option('excludeSwitches', ['enable-logging'])

    driver = webdriver.Chrome(options=option)

    demo_form_submit(driver)