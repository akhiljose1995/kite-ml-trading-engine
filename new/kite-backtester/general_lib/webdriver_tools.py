from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import config

class WebDriverSession:
    def __init__(self, headless=False):
        self.driver = self._init_driver(headless)

    def _init_driver(self, headless):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--start-maximized")
        return webdriver.Chrome(options=options)
    
    def get_driver(self):
        return self.driver

    def login_zerodha(self):
        self.driver.get("https://kite.zerodha.com")
        self.driver.find_element(By.ID, "userid").send_keys(config.USER_NAME)
        self.driver.find_element(By.ID, "password").send_keys(config.PASSWORD)
        self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        input("Press Enter after completing 2FA in browser...")

    def wait_for_element(self, selector: str, by=By.CSS_SELECTOR, timeout=10):
        WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((by, selector))
        )

    def get_soup(self):
        return BeautifulSoup(self.driver.page_source, "html.parser")

    def quit(self):
        self.driver.quit()
