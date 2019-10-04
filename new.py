import array
import csv
import math

import minecart
import cv2
import numpy as np
import pytesseract as pytesseract
from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import argparse
import pandas as pd
DEBUG = False
ROW_MIN_HEIGHT = 5 #in px
COLUMN_MIN_WIDTH = 5
PADDING = 2


def process_file(filepath):
    filename=filepath.split("/")[1]

    pdffile = open(filepath, 'rb')
    doc = minecart.Document(pdffile)

    with open("output/"+filename +'.csv', 'w') as csvfile:
     with open( "output/"+filename+"_table_data.txt",'w')as text_file:
      with open("output/"+filename+"whole_without_table.txt","w",encoding="utf8")as whole_file:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        pagen = 0
        #iterating through all pages
        for page in doc.iter_pages():
            pagen += 1
            if len(page.images) == 0:
                print("Page {}: No Images found".format (pagen))
                continue
            im = page.images[0].as_pil()  # requires
            text = pytesseract.image_to_string(im)
            whole_file.write(text)
            print(text)
            im = im.convert('L') #validate grayscale
            gray_image = np.array(im)
            extracted_table = extract_main_table(gray_image)
            if DEBUG:
             show_wait_destroy("extracted",extracted_table)
            row_images = extract_rows_columns(extracted_table) #[1:]
            if len(row_images) == 0:
                continue

            idx = 0
            for row in row_images:
                idx += 1
                print("{0} : Extracting row {1} out of {2} page {3}" .format(filename, idx,len(row_images), pagen))
                row_texts = []
                for column in row:
                    text = pytesseract.image_to_string(column)
                    row_texts.append(text)
                text_data=" |".join(row_texts)
                text_file.write(text_data)
                text_file.write("\n")
                csv_writer.writerow(row_texts)

    df=pd.read_csv('output/'+filename +'.csv')

    print(df.columns)
    xtrain = df.iloc[:,[1]]
    # print(xtrain)
    debit=df[pd.isnull(df.iloc[:,[2]]).any(axis=1)].values.tolist()

    credit=df[pd.isnull(df.iloc[:,[3]]).any(axis=1)].values.tolist()
    debit=debit.dropna()
    print(debit)



    # # print(credit)
    # with open("final_output.txt","w") as of:
    #   of.write("credits\n")
    #   of.write(join_credits)
    #   of.write("\n")
    #   of.write("debits\n")
    #   of.write(join_debits)

def extract_main_table(gray_image):
    inverted = cv2.bitwise_not(gray_image)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

    thresholded = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if DEBUG:
        show_wait_destroy("thresholded",thresholded)

    cnts = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]# if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    extracted = four_point_transform(gray_image.copy(), box.reshape(4, 2))

    if DEBUG:
        color_image = cv2.cvtColor(gray_image.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(color_image,[box],0,(0,0,255),2)
        cv2.drawContours(color_image, [cnts[0]], -1, (0, 255, 0), 2)


    return extracted

def horizontal_boxes_filter(box,width):
    x,y,w,h = box
    return w > width * 0.7

def vertical_boxes_filter(box,height):
    x,y,w,h = box
    return  h > height * 0.7


def extract_rows_columns(gray_image):
    inverted = cv2.bitwise_not(gray_image)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

    height, width = gray_image.shape

    thresholded = cv2.threshold(blurred, 128, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    vertical_kernel_height = math.ceil(height*0.3)
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height))

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    horizontal_kernel_width = math.ceil(width*0.3)
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(thresholded, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    _, vertical_contours, _ = cv2.findContours(verticle_lines_img.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (vertical_contours, vertical_bounding_boxes) = sort_contours(vertical_contours, method="left-to-right")

    filtered_vertical_bounding_boxes = list(filter(lambda x:vertical_boxes_filter(x,height), vertical_bounding_boxes))




    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(thresholded, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    _, horizontal_contours, _ = cv2.findContours(horizontal_lines_img.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    horizontal_contours, horizontal_bounding_boxes = sort_contours(horizontal_contours, method="top-to-bottom")

    filtered_horizontal_bounding_boxes = list(filter(lambda x:horizontal_boxes_filter(x,width), horizontal_bounding_boxes))

    if DEBUG:
        color_image = cv2.cvtColor(gray_image.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(color_image, vertical_contours, -1, (0, 255, 0), 2)
        cv2.drawContours(color_image, horizontal_contours, -1, (255, 0, 0), 2)

       

        show_wait_destroy("horizontal_vertical_contours",color_image)


    extracted_rows_columns = []



    for idx_h, horizontal_bounding_box in enumerate(filtered_horizontal_bounding_boxes):
        if idx_h == 0:
            continue
        hx_p,hy_p,hw_p,hh_p = filtered_horizontal_bounding_boxes[idx_h-1] #previous horizontal box
        hx_c,hy_c,hw_c,hh_c = horizontal_bounding_box

        extracted_columns = []
        for idx_v, vertical_bounding_box in enumerate(filtered_vertical_bounding_boxes):
            if idx_v == 0:
                continue
            vx_p,vy_p,vw_p,vh_p = filtered_vertical_bounding_boxes[idx_v-1] #previous horizontal box
            vx_c,vy_c,vw_c,vh_c = vertical_bounding_box
            table_cell = gray_image[hy_p:hy_c+hh_c,vx_p:vx_c+vw_c]



            blurred = cv2.GaussianBlur(table_cell, (5, 5), 0)
            #cv2.rectangle(color_image,(vx_p,hy_p),(vx_c+vw_c,hy_c+hh_c),(255,0,0),2)

            thresholded = cv2.threshold(blurred, 128, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            im2, contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            extracted = four_point_transform(table_cell.copy(), box.reshape(4, 2))[1:-1,1:-1] #remove 1 px from each side
            ret,extracted = cv2.threshold(extracted,165,255,cv2.THRESH_BINARY)
            extracted_columns.append(extracted)

            # cv2.drawContours(color_image, [contours[0]], -1, (0,255,0), 3)


        extracted_rows_columns.append(extracted_columns)

    return extracted_rows_columns



def show_wait_destroy(winname, img):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)
    cv2.resizeWindow(winname, 1000,800)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)
filepath="pdf/demo2.pdf"
process_file(filepath)



