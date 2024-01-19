#compressing certificates
import cv2
# from jupyter_client import KernelConnectionInfo
import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import requests
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your actual path

# Function to convert text to image
def text_to_image(text, image_size=(256, 256), font_size=20):
    image = Image.new("RGB", image_size, (0,0,0,0))
    draw = ImageDraw.Draw(image)
    # font = ImageFont.load_default()
    font = ImageFont.truetype("arial.ttf", font_size)  # You may need to adjust the font file path

    # font=40
     # Get the bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    
    # Calculate text width and height
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.array(image)

# URL of the watermark text
watermark_url = 'phase'
watermark_text = watermark_url

# Convert the text to an image
watermark_image = text_to_image(watermark_text,font_size=20)

cv2.imshow("original watermark", watermark_image)
cv2.imwrite("text_image.png", watermark_image)
# watermark_image=cv2.imread('text_image.png', cv2.IMREAD_GRAYSCALE)
# Convert the watermark_image to float32 and grayscale
watermark_image_float32 = np.float32(cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY))
# watermark_image_float32 = np.float32(cv2.imread('text_image.png', cv2.IMREAD_GRAYSCALE))

# watermark_image_float32 = np.float32(watermark_image)

# Apply DCT to the watermark image
watermark_dct = cv2.dct(watermark_image_float32)

# Zero out some coefficients in the watermark DCT
rows, cols = watermark_dct.shape
for i in range(rows):
    for j in range(cols - i, cols):
        watermark_dct[i, j] = 0


# Store the dimensions of the original image
# new_img=cv2.resize('certificate.png',(1024,1024))

host_image1 = cv2.imread('certificate.png', cv2.IMREAD_GRAYSCALE)

compression_quality = 90

# Compress the image and save it as JPEG
cv2.imwrite('compressed_host_image.jpg', host_image1, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])
host_image=cv2.imread('compressed_host_image.jpg',cv2.IMREAD_GRAYSCALE)
original_height, original_width= host_image.shape

host_image=cv2.resize(host_image,(1024,1024))

wavelet = 'haar'
coeffs = pywt.dwt2(host_image, wavelet)
LL, (LH, HL, HH) = coeffs
LL2, (LH2, HL2, HH2) = pywt.dwt2(HH, wavelet)
# cv2.imshow("watermark dct",watermark_dct)
cv2.imshow("HH 2nd band", HH2)
# embedding into hh band
for i in range(rows):
    for j in range(0, cols-i):
        HH2[i,j]=watermark_dct[i, j]

# Embed the watermark into the HH subband of the host image
coeffs_2= (LL2, (LH2, HL2, HH2))
cv2.imshow("HH 2nd embedding", HH2)

# watermarked_hh= pywt.idwt2((LL2, (LH2, HL2, HH2)),wavelet)
watermarked_hh = pywt.idwt2(coeffs_2, wavelet)

watermarked_image= pywt.idwt2((LL, (LH, HL, watermarked_hh)),wavelet)
# LLi2,(LHi2, HLi2, HHi2)= pywt.idwt2(coeffs_2,wavelet)
# watermarked_image= pywt.idwt2(coeffs,wavelet)

watermarked_image = watermarked_image/255 


# watermarked_image = host_image.copy()
# watermarked_image[:,:,0] = watermarked_image


# cv2.imshow("watermaked_image",watermarked_image)
# watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
# watermarked_image = watermarked_image/ 255.0
cv2.imshow("watermaked_image",cv2.resize(watermarked_image,(512, 512)))
# watermarked_image_float32_cert = np.float32(cv2.cvtColor(watermarked_image, cv2.COLOR_GRAY2BGR))
# watermark_image_float32 = np.float32(cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY))

# watermarked_image_color = cv2.merge([watermarked_image*255, watermarked_image*255, watermarked_image*255])
# watermarked_image_color_resized = cv2.resize(watermarked_image_color, (original_width, original_height))
# cv2.imwrite("watermaked_image_dbl_dwt_cert.png", watermarked_image_color_resized)

cv2.imwrite("watermaked_image_dbl_dwt_cert_compress.png",cv2.resize(watermarked_image*255,(original_width,original_height)))

#Apply DWT to the watermarked image
coeffs_w = pywt.dwt2(watermarked_image, wavelet)
LL_w, (LH_w, HL_w, HH_w) = coeffs_w

LL2_w, (LH2_w, HL2_w, HH2_w) = pywt.dwt2(HH_w, wavelet)
# print(HH_w.size)
# cv2.imshow("HH",HH_w*255)
# Normalize the watermarked image to the range [0, 1]
# cv2.imshow("watermaked_image",watermarked_image)



row_hh,cols_hh=HH2_w.shape
extr_wt=np.zeros_like(HH2_w)
for i in range(0, row_hh):
    for j in range(cols_hh-i,cols_hh):
           HH2_w[i, j]=0


# cv2.imshow("extr_after",extr_wt*255)
cv2.imshow("HH extracting ",HH2_w)
# Extract the watermark from the HH subband
extr = cv2.idct(HH2_w)
# extr=extr*255
cv2.imshow("extracted watermark",cv2.resize(extr,(256,256)))
cv2.imwrite("text_image.png",extr)

# cv2.imshow('Extracted Watermark', (extracted_watermark * 255).astype(np.uint8))
# cv2.imshow('Reconstructed Watermarked Image', (reconstructed_watermarked_image * 255).astype(np.uint8))

image_path = 'text_image.png'

# extracted_text = image_to_text(image_path)
extracted_text = pytesseract.image_to_string(image_path, lang='eng', config='--psm 6')
print("Extracted Text:", extracted_text)

cv2.waitKey(0)
cv2.destroyAllWindows()