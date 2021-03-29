import os, cv2, pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm 

DIR =  '.'
img_dir = 'imgs'
txt_dir = 'texts'
exceptions = []#'0001.pdf']
try:
	os.mkdir(img_dir)
except:
	pass

pdfs = os.listdir(DIR)
# pdfs = ['Scientific2.pdf', 'Thesis4.pdf', 'Thesis5.pdf', 'Thesis6.pdf', 'Thesis7.pdf', 'Thesis8.pdf']
for pdf in pdfs:
	if pdf in exceptions :
			continue
	if pdf.endswith('.pdf'):
		file_name = pdf[:-4]
		print(file_name)
		new_dir = os.path.join(img_dir,file_name)
		new_txt_dir = os.path.join(txt_dir,file_name)
		try:
			os.mkdir(new_dir)
		except:
			pass
		try:
			os.mkdir(new_txt_dir)
		except:
			pass
		# break
		pages = convert_from_path(pdf, output_folder=new_dir, fmt='jpg')

		pages = os.listdir(new_dir)
		for page in tqdm(pages):
			img_path = os.path.join(new_dir,page)
			txt_path = img_path.replace('imgs','texts')[:-4]+".txt"
			# pdb.set_trac e()
			img = cv2.imread(img_path,cv2.IMREAD_COLOR)
			text = pytesseract.image_to_string(img)
			f = open(txt_path, "w")
			f.write(text)
			f.close()

		# print('...')
		# for i, page in enumerate(pages):
		# 	page.save(img_dir+'\\'+pdf[-4]+'_'+str(i)+'.jpg', 'JPEG')
		# break