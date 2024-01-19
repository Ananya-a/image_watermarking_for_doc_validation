from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import cv2

#for clr 
# original_image = cv2.imread('Lenna.png')
# watermarked_image = cv2.imread('watermarked_image_YCbCr_LH_lenna.jpg')
# original_image_ycc = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)
# watermarked_image_ycc = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
# original_image_y = original_image_ycc[:, :, 0]
# watermarked_image_y = watermarked_image_ycc[:, :, 0]
# ssim_value, _ = compare_ssim(original_image_y, watermarked_image_y, full=True)
# print(f"SSIM Value: {ssim_value}")

#grey scale 
original_image = cv2.imread('certificate.png', cv2.IMREAD_GRAYSCALE)
watermarked_image=cv2.imread('watermaked_image_dbl_dwt_cert_compress.png',cv2.IMREAD_GRAYSCALE)
# Calculate SSIM
ssim_score = compare_ssim(original_image, watermarked_image)
print("SSIM Value:", ssim_score)




def psnr(original, reconstructed):
    # Assuming the images are in grayscale
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel_value = 255.0
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr_value

# Read the original and reconstructed images
# original_image = cv2.imread('hoo.jpeg', cv2.IMREAD_GRAYSCALE)
# reconstructed_image = cv2.imread('extracted_watermark_HH_hoo.jpg', cv2.IMREAD_GRAYSCALE)
# original_image = cv2.imread('Lenna.png')
# reconstructed_image = cv2.imread('watermarked_image_CLR_HH_lenna.jpg')



# Check if images are loaded successfully
if original_image is None or watermarked_image is None:
    print("Error: Unable to read images.")
else:
    # Ensure images have the same size
    if original_image.shape == watermarked_image.shape:
        # Calculate PSNR
        psnr_value = psnr(original_image, watermarked_image)
        print(f"PSNR Value: {psnr_value} dB")
    else:
        print("Error: Images have different dimensions.")