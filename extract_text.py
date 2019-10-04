# Import libraries 
from PIL import Image 
import pytesseract 
import sys 
from pdf2image import convert_from_path 
import os 
import PIL.Image  

# Path of the pdf 
PDF_file = "img3.pdf"
# config='-psm 6'

''' 
Part #1 : Converting PDF to images 
'''

# convert all the pages of the PDF into variable 
pages = convert_from_path(PDF_file, 500) 
count = 1

# Iterate through all the pages stored above 
for page in pages: 

	filename = "page"+str(count)+".jpg"
	
	# Save the image of the page in system 
	page.save(filename, 'JPEG') 

	# Increment the counter to update filename 
	count = count + 1

 
filelimit = count-1

# Creating a text file to write the output 
outfile_text= "out_text.txt"

# Open the file in append mode so that 
# All contents of all images are added to the same file 
f = open(outfile_text, "a") 

# Iterate from 1 to total number of pages 
for i in range(1, filelimit + 1): 

	# Set filename to recognize text from 
	# Again, these files will be: 
	# page_1.jpg 
	# page_2.jpg 
	# .... 
	# page_n.jpg 
	filename = "page"+str(i)+".jpg"
		
	# Recognize the text as string in image using pytesserct 
	text = str(((pytesseract.image_to_string(Image.open(filename)))))
        


	# The recognized text is stored in variable text 
	# Any string processing may be applied on text 
	# Here, basic formatting has been done: 
	# In many PDFs, at line ending, if a word can't 
	# be written fully, a 'hyphen' is added. 
	# The rest of the word is written in the next line 
	# Eg: This is a sample text this word here GeeksF- 
	# orGeeks is half on first line, remaining on next. 
	# To remove this, we replace every '-\n' to ''. 
	text = text.replace('-\n', '')	 

	# Finally, write the processed text to the file. 
	f.write(text) 

# Close the file after writing all the text. 
f.close() 
