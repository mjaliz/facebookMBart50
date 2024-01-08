from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager

# options = Options()
# options.binary = FirefoxBinary("/usr/bin/firefox")

# driver = webdriver.Firefox(GeckoDriverManager().install())
driver = webdriver.Chrome()
driver.get("https://dic.b-amooz.com/en/dictionary")
search_elem = driver.find_element(By.ID, "search-input")
search_elem.clear()
search_elem.send_keys("apple")
search_elem.send_keys(Keys.ENTER)
