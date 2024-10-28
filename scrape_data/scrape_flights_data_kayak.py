from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException
from datetime import date, timedelta
import pandas as pd
import time
import os

# Define global driver to prevent repetition in functions
driver = None

def init_driver():
    global driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.maximize_window()

def get_url(departure_date, place):
    url = (
        f"https://aertrip.com/flights?return={departure_date}&adult=1&child=0&infant=0&trip_type=return"
        f"&totalLegs=2&origin=BOM&depart={departure_date}&destination={place}&cabinclass=Economy"
        "&nonStopFlag=true"  # Ensure non-stop flag is enabled
    )
    return url

def scroll_bottom():
    SCROLL_PAUSE_TIME = 1
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def click_load_all_button():
    buttons = driver.find_elements(By.CLASS_NAME, 'css-total-fl-text')
    for button in buttons:
        try:
            button.click()
            time.sleep(1)  # Give it some time to load
        except Exception as e:
            print(f"Error clicking load button: {e}")

def toggle_places():
    try:
        toggle_button = driver.find_element(By.CLASS_NAME, 'interchange')
        toggle_button.click()
    except NoSuchElementException:
        print("Toggle button not found.")
    except Exception as e:
        print(f"Toggle error: {e}")

def click_search_button():
    try:
        button = driver.find_element(By.CLASS_NAME, 'buttonDiv')
        button.click()
        time.sleep(10)
    except NoSuchElementException:
        print("Search button not found.")
    except Exception as e:
        print(f"Search button click error: {e}")

def get_flights(max_retries=3):
    # Wait to ensure elements are loaded
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'departureDetailsBox')))
    except TimeoutException:
        print("Flight elements took too long to load.")

    flights = driver.find_elements(By.CLASS_NAME, 'departureDetailsBox')
    if not flights:
        print("No flights found.")

    # Collect flight data with retry logic for stale elements
    flight_data = []
    for flight in flights:
        retries = 0
        while retries < max_retries:
            try:
                # Attempt to collect flight text data
                flight_data.append(flight.text.split('\n'))
                break  # Exit loop if successful
            except StaleElementReferenceException:
                retries += 1
                if retries >= max_retries:
                    print("Skipping flight due to repeated stale reference issues.")
                else:
                    print(f"Retrying stale element (attempt {retries})...")

    return flight_data


def get_df(flights):
    columns = ['Origin', 'Destination', 'Company', 'Departure Time', 'Arrival Time', 'Duration Time', 'Flight Price']
    df = pd.DataFrame(columns=columns)

    for f in flights:
        print(f)  # Debug: print flight data to verify structure
        
        # Skip flights with layover information
        if any("Layover" in part for part in f):
            print("Skipping flight with layover:", f)
            continue

        try:
            # Extract origin and destination codes from the 5th element (formatted as "BOMHYD")
            origin, destination = f[4][:3], f[4][-3:]  
            
            # Check if origin and destination are the same
            if origin == destination:
                print(f"Skipping flight with same origin and destination: {origin} to {destination}")
                continue
            
            # Extract remaining details
            company = f[1].split('-')[0].strip()
            departure_time = f[0].strip()
            arrival_time = f[2].split()[0] if f[2].endswith('+1') else f[2].strip()
            duration_time = f[3].strip()
            flight_price = f[-1].strip() if f[-1].replace(",", "").isdigit() else None
            
            # Only include entries with a valid price
            if flight_price:
                df = pd.concat([df, pd.DataFrame([{
                    'Origin': origin,
                    'Destination': destination,
                    'Company': company,
                    'Departure Time': departure_time,
                    'Arrival Time': arrival_time,
                    'Duration Time': duration_time,
                    'Flight Price': flight_price
                }])], ignore_index=True)
        except IndexError:
            print("Flight data not in expected format:", f)
    
    return df


# Define airport codes
airports = {
    #'BLR': 'Bengaluru, Kempegowda International Airport',
    'CCU': 'Kolkata, Netaji Subhas Chandra Bose Airport',
    'MAA': 'Chennai, Chennai Airport',
    'DEL': 'Delhi, Indira Gandhi International Airport',
    'HYD': 'Hyderabad, Rajiv Gandhi International Airport'
}

# Generating dates
airport_codes = list(airports.keys())
today = date.today()
list_of_dates = [(today + timedelta(days=i)).strftime('%d-%m-%Y') for i in range(7, 22)]
print(list_of_dates)

# Creating directories for storing data
original = os.getcwd()
for place in airport_codes:
    os.makedirs(place, exist_ok=True)

# Main scraping
init_driver()
for place in airport_codes:
    for departure_date in list_of_dates:
        url = get_url(departure_date, place)
        driver.get(url)
        
        try:
            WebDriverWait(driver, 20).until(EC.invisibility_of_element_located((By.ID, 'flight-search-loader')))
        except TimeoutException:
            print("Loading took too much time! Loader did not disappear.")
        
        scroll_bottom()
        click_load_all_button()
        
        flights = get_flights()
        df = get_df(flights)
        toggle_places()
        click_search_button()
        
        try:
            WebDriverWait(driver, 20).until(EC.invisibility_of_element_located((By.ID, 'flight-search-loader')))
        except TimeoutException:
            print("Loading took too much time! Loader did not disappear.")
        
        scroll_bottom()
        click_load_all_button()
        
        flights = get_flights()
        df_additional = get_df(flights)
        df = pd.concat([df, df_additional], ignore_index=True)
        
        df['Date'], df['Cabin Class'] = departure_date, 'Economy'
        df.to_csv(os.path.join(place, f"{departure_date}.csv"), index=False)
        
driver.quit()

# Compile data
main_df = pd.DataFrame()
for place in airport_codes:
    place_df = pd.DataFrame()
    for departure_date in list_of_dates:
        df = pd.read_csv(os.path.join(place, f"{departure_date}.csv"))
        place_df = pd.concat([place_df, df], ignore_index=True)
    place_df.to_csv(f"{place}.csv", index=False)
    main_df = pd.concat([main_df, place_df], ignore_index=True)

main_df.to_csv('data.csv', index=False)


#'BOM': 'Mumbai, Chhatrapati Shivaji International Airport',

    


# from time import sleep
# import pandas as pd
# from selenium import webdriver
# from bs4 import BeautifulSoup
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# chrome_options = Options()
# chrome_options.add_argument("--no-proxy-server")
# driver = webdriver.Chrome(options=chrome_options)

# to_location = 'BLR'
# url = f'https://www.kayak.co.in/flights/IXC-{to_location}/2024-11-08?sort=bestflight_a'

# driver.get(url)

# # Wait for the elements to be present
# try:
#     WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.XPATH, '//div[@class="nrc6-inner"]'))
#     )
# except Exception as e:
#     print("Error:", e)
#     driver.quit()

# # Corrected syntax for find_elements
# flight_rows = driver.find_elements(By.XPATH, '//div[@class="nrc6-inner"]')

# lst_prices = []
# lst_company_names = []

# # Extract HTML for each flight row
# for WebElement in flight_rows:
#     elementHTML = WebElement.get_attribute('outerHTML')
#     elementSoup = BeautifulSoup(elementHTML, 'html.parser')

#     # Try to find the price section and price text
#     temp_price = elementSoup.find("div", {"class": "nrc6-price-section"})
#     if temp_price:
#         price = temp_price.find("div", {"class": "f8F1-price-text"})
#         if price:
#             lst_prices.append(price.text.replace('₹', '').replace(',', '').strip())  # Clean price for numeric conversion
#         else:
#             print("Price text not found")
#     else:
#         print("Price section not found")

#     # Find company/airline name section
#     # temp_name = elementSoup.find("div", {"class": "ksmO-content-wrapper"})
#     # if temp_name:
#     #     name = temp_name.find("div")  # Grabbing the nested div containing the airline name
#     #     if name:
#     #         lst_company_names.append(name.text.strip())  # Extracting and cleaning the text
#     #     else:
#     #         print("Company name text not found")
#     # else:
#     #     print("Company name section not found")

# # Optional: Add sleep if needed
# sleep(5)

# # Don't forget to close the driver at the end
# driver.quit()

# # Creating a DataFrame to store the data
# data = {
#     #'Company': lst_company_names,
#     'Price': [int(price) for price in lst_prices]  # Convert prices to integers
# }

# df = pd.DataFrame(data)

# # Display the DataFrame
# print(df)

# # Optional: Save DataFrame to a CSV file
# df.to_csv('flight_prices.csv', index=False)
