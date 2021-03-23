from google_images_search import GoogleImagesSearch
import sys
import time
import argparse
import math
from ng_secret_keys import *
import os

def crawl_google_search_images (pattern = None, num = 200, width = 500, height = 500, outdir="ng_output_images") : 
    if pattern is None : 
        print ("ERROR : No search pattern ")
        return 0

    pattern_list = pattern.split(",")
    for p_pre in pattern_list :

        p = p_pre.strip()

        print ("Getting {} images of {} ...".format(num, p))

        # you can provide API key and CX using arguments,
        # or you can set environment variables: GCS_DEVELOPER_KEY (API Key), GCS_CX (search engine ID)
        gis = GoogleImagesSearch(API_KEY, CX_ID)
    
        # define search params:
            #'imgSize': 'huge|icon|large|medium|small|xlarge|xxlarge',
            #'imgDominantColor': 'BLACK|BLUE|BROWN|GRAY|GREEN|PINK|PURPLE|TEAL|WHITE|YELLOW',
            #'rights': 'CC_PUBLICDOMAIN|CC_ATTRIBUTE|CC_SHAREALIKE|CC_NONCOMMERCIAL|CC_NONDERIVED'
        _search_params = {
            'q': p,
            'num': num,
            'fileType': 'jpg',
            'imgType': 'photo',
            'imgSize': 'LARGE',
            'rights': 'CC_PUBLICDOMAIN'
        }
        
        output_dir = outdir + str(os.path.sep) + p.replace(" ", "_")
        #gis.search(search_params=_search_params,
        #        path_to_dir = output_dir,
        #        custom_image_name=p.replace(" ", "_"),
        #        height = height,
        #        width = width)
        #for image in gis.results() :
        #    image.download(output_dir)
        try : 
            gis.search(search_params=_search_params,
                    path_to_dir = output_dir,
                    custom_image_name=p.replace(" ", "_"),
                    height = height,
                    width = width)
        except : 
            print("Unexpected error:", sys.exc_info()[0])
            pass

def run (pattern, num, height, width, outdir):
    result = crawl_google_search_images (pattern, num, height, width, outdir) 

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--pattern', help='search pattern')
    ap.add_argument('-n', '--number', type=int, default=1000, help='the number of result images')
    ap.add_argument('-o', '--output', default="ng_output_images", help='output directory')
    ap.add_argument('--height', type=int, default=100, help='the width of output image')
    ap.add_argument('--width', type=int, default=100, help='the height of output image')
    args = vars(ap.parse_args())
    run(args['pattern'], args['number'], args['height'], args['width'], args['output'])
