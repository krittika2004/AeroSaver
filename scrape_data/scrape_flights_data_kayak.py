from time import sleep
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options = Options()
chrome_options.add_argument("--no-proxy-server")
driver = webdriver.Chrome(options=chrome_options)

to_location = 'BLR'
url = f'https://www.kayak.co.in/flights/IXC-{to_location}/2024-11-08?sort=bestflight_a'

driver.get(url)

# Wait for the elements to be present
try:
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//div[@class="nrc6-inner"]'))
    )
except Exception as e:
    print("Error:", e)
    driver.quit()

# Corrected syntax for find_elements
flight_rows = driver.find_elements(By.XPATH, '//div[@class="nrc6-inner"]')

lst_prices = []
lst_company_names = []

# Extract HTML for each flight row
for WebElement in flight_rows:
    elementHTML = WebElement.get_attribute('outerHTML')
    elementSoup = BeautifulSoup(elementHTML, 'html.parser')

    # Try to find the price section and price text
    temp_price = elementSoup.find("div", {"class": "nrc6-price-section"})
    if temp_price:
        price = temp_price.find("div", {"class": "f8F1-price-text"})
        if price:
            lst_prices.append(price.text.replace('₹', '').replace(',', '').strip())  # Clean price for numeric conversion
        else:
            print("Price text not found")
    else:
        print("Price section not found")

    # Find company/airline name section
    # temp_name = elementSoup.find("div", {"class": "ksmO-content-wrapper"})
    # if temp_name:
    #     name = temp_name.find("div")  # Grabbing the nested div containing the airline name
    #     if name:
    #         lst_company_names.append(name.text.strip())  # Extracting and cleaning the text
    #     else:
    #         print("Company name text not found")
    # else:
    #     print("Company name section not found")

# Optional: Add sleep if needed
sleep(5)

# Don't forget to close the driver at the end
driver.quit()

# Creating a DataFrame to store the data
data = {
    #'Company': lst_company_names,
    'Price': [int(price) for price in lst_prices]  # Convert prices to integers
}

df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Optional: Save DataFrame to a CSV file
df.to_csv('flight_prices.csv', index=False)
