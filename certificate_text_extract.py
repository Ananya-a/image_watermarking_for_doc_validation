#extracting text from certificates
import cv2
# from jupyter_client import KernelConnectionInfo
import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import requests

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your actual path

# host = cv2.imread('watermaked_image_dbl_dwt.png', cv2.IMREAD_GRAYSCALE)
host = cv2.imread('watermaked_image_dbl_dwt_cert_compress.png', cv2.IMREAD_GRAYSCALE)
watermarked_image = cv2.resize(host, (1024, 1024))


# Save the resized image
# cv2.imwrite('certificate1024.png', host_image)
wavelet = 'haar'
coeffs_w = pywt.dwt2(watermarked_image, wavelet)
LL_w, (LH_w, HL_w, HH_w) = coeffs_w

LL2_w, (LH2_w, HL2_w, HH2_w) = pywt.dwt2(HH_w, wavelet)
# cv2.imshow("watermaked_image",watermarked_image)


row_hh,cols_hh=HH2_w.shape
extr=np.zeros_like(HH2_w)
for i in range(0, row_hh):
    for j in range(cols_hh-i,cols_hh):
           HH2_w[i, j]=0

normalized_watermark=HH2_w/255

# cv2.imshow("extr_after",extr_wt*255)
# cv2.imshow("HH extracting ",HH2_w)
# Extract the watermark from the HH subband
extr = cv2.idct(normalized_watermark)
# extr = cv2.idct(HH2_w)
# extr=extr*255
cv2.imshow("extracted watermark",cv2.resize(extr,(256,256)))
cv2.imwrite("extracted_text_image.png",extr*255)
image_path='extracted_text_image.png'
# extracted_text = image_to_text(image_path)
extracted_text = pytesseract.image_to_string(image_path, lang='eng', config='--psm 6')
print("Extracted Text:", extracted_text)

cv2.waitKey(0)
cv2.destroyAllWindows()