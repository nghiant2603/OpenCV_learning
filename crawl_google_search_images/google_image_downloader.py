from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import urllib.request
import time
import sys
import os
import argparse

def crawl_google_images(pattern = None, output_dir = "ng_images", browser_driver="ng_chromedriver.exe", limit = 1000) : 

    if pattern is None : 
        print("[ERROR] : There is no search pattern. Please specify it...")
    else : 
        pattern_list = pattern.split(",")
        for j in pattern_list : 
            site = 'https://www.google.com/search?tbm=isch&q=' + j.strip()

            seearch_query = j.strip().replace(" ", "_")
            outdir = output_dir + os.path.sep + seearch_query
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            #providing driver path
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')  # Last I checked this was necessary.
            driver = webdriver.Chrome(browser_driver, chrome_options=options)
            #driver = webdriver.Chrome(executable_path = browser_driver)

            #passing site url
            driver.get(site)

            #if you just want to download 10-15 images then skip the while loop and just write
            #driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")

            #below while loop scrolls the webpage 7 times(if available)
            i = 0
            while i<10:  
            	#for scrolling page
                driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")
                try : 
                    #driver.findElement(By.value("Show more results"))
                    NEXT_BUTTON_XPATH = '//input[@type="button" and @value="Show more results"]'
                    driver.find_element_by_xpath(NEXT_BUTTON_XPATH).click()
                except Exception as e:
                    pass
                time.sleep(5)
                i+=1

            #parsing
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            #closing web browser
            driver.close()
            #scraping image urls with the help of image tag and class used for images
            img_tags = soup.find_all("img", class_="rg_i")

            print ("Downloading " + str(seearch_query) + " images...")
            n = 0
            for i in img_tags:
                try:
                    if n < limit : 
            		    #passing image urls one by one and downloading
                        urllib.request.urlretrieve(i['src'], outdir + os.path.sep + seearch_query + str(n) + ".jpg")
                        n += 1
                    else : 
                        break
                except Exception as e:
                    pass
            print("Number of downloaded images : " + str(n))
            print("Total found images : " + str(len(img_tags)))

def run (pattern, outdir, browser_driver, limit):
    crawl_google_images (pattern, outdir, browser_driver, limit) 

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--pattern', help='search pattern')
    ap.add_argument('-o', '--output', default="ng_images", help='output directory')
    ap.add_argument('-d', '--driver', default="ng_chromedriver", help='chrome driver')
    ap.add_argument('-n', '--limit', type=int, default=1000, help='The number of downloaded images')
    args = vars(ap.parse_args())
    run(args['pattern'], args['output'], args['driver'], args['limit'])
