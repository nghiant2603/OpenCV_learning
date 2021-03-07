from google_images_search import GoogleImagesSearch
import sys
import time
import argparse
import math
import secret_keys
import os

def crawl_google_search_images (pattern = None, num = 1000) : 
    if pattern is None : 
        print ("ERROR : No search pattern ")
        return 0

    # you can provide API key and CX using arguments,
    # or you can set environment variables: GCS_DEVELOPER_KEY (API Key), GCS_CX (search engine ID)
    gis = GoogleImagesSearch(secret_keys.API_KEY, secret_keys.CX_ID)
    
    # define search params:
    _search_params = {
        'q': pattern,
        'num': 100,
        #'safe': 'off',
        'fileType': 'jpg|gif|png',
        'imgType': 'photo',
        #'imgSize': 'huge|icon|large|medium|small|xlarge|xxlarge',
        'imgSize': 'LARGE',
        #'imgDominantColor': 'BLACK|BLUE|BROWN|GRAY|GREEN|PINK|PURPLE|TEAL|WHITE|YELLOW',
        #'rights': 'CC_PUBLICDOMAIN|CC_ATTRIBUTE|CC_SHAREALIKE|CC_NONCOMMERCIAL|CC_NONDERIVED'
    }
    
    # this will only search for images:
    #gis.search(search_params=_search_params)
    
    # this will search and download:
    #gis.search(search_params=_search_params, path_to_dir='./google_images/')
    
    # this will search, download and resize:
    epoch = math.ceil(num/100.0)
    output_dir = './google_images/{}'.format(pattern)

    for i in range(epoch) :
        print("[INFO] Epoch : ", i)
        tmp = time.time()
        gis.search(search_params=_search_params,
                path_to_dir = output_dir,
                custom_image_name=pattern)
        print("\tFinish epoch : ", i, " - Time : ", time.time() - tmp)

    path, dirs, files = next(os.walk(output_dir)) 
    while (len(files) < num) : 
        print("Get more images...")
        gis.search(search_params=_search_params,
                path_to_dir = output_dir,
                custom_image_name=pattern)
        path, dirs, files = next(os.walk(output_dir)) 
        time.sleep(50)

    # search first, then download and resize afterwards:
    #gis.search(search_params=_search_params)
    #for image in gis.results():
    #    image.download('./google_images/')
    #    image.resize(500, 500)

def run (pattern, num):
    print ("Getting {} images of {} ...".format(num, pattern))
    result = crawl_google_search_images (pattern, num) 

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--pattern', help='searc pattern')
    ap.add_argument('-n', '--number', type=int, default=1000, help='the number of result images')
    args = vars(ap.parse_args())
    run(args['pattern'], args['number'])
