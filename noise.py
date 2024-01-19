import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
# def add_gaussian_noise(image, stddev):
#     noise = np.zeros_like(image, dtype=np.uint8)
#     cv2.randn(noise, 128, stddev)
#     noisy_image = cv2.addWeighted(image, 1.0, noise, 0.5, 0)
#     return noisy_image
def apply_gaussian_filter(image, sigma):
    return cv2.GaussianBlur(image, (5, 5), sigma)

def addition_noise(img_0,num_pixels):
    rw, cl=img_0.shape # shape gives no of rows and columns
    # num_pixels=random.randint(300,10000)
    for i in range(num_pixels):
        #salt section
        y_crd=random.randint(0,rw-1)
        x_crd=random.randint(0,cl-1)
        img_0[y_crd][x_crd]=255   # throw white dots on the pixel
    for i in range(num_pixels):
        #pepper section
        y_crd=random.randint(0,rw-1)
        x_crd=random.randint(0,cl-1)
        img_0[y_crd][x_crd]=0  # black  
        
    return(img_0)
def add_poisson_noise(image):
    noise = np.random.poisson(image.astype(float) / 255.0) * 255
    noisy_image = cv2.add(image, noise.astype(np.uint8))
    return noisy_image

def save_image(image, filename):
    cv2.imwrite(filename, image)
# Load an image from file
image_path = 'certificate_text_embed.png'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.subplot(2, 4, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# sigma_values = [10, 20, 30, 40]
# for i, sigma in enumerate(sigma_values):
#     plt.subplot(2, 4, i + 5)
#     noisy_image = add_gaussian_noise(original_image, sigma)
#     plt.imshow(noisy_image, cmap='gray')
#     plt.title(f'Sigma = {sigma}')
#     plt.axis('off')
#     image_filename = f'gaussian_noisy_sigma_{sigma}.png'
#     cv2.imwrite(image_filename, noisy_image)

sigma_values = [0.5, 1.0, 1.5]
filtered_images = [apply_gaussian_filter(original_image, sigma) for sigma in sigma_values]

# Save the filtered images
for i, sigma in enumerate(sigma_values):
    save_image(filtered_images[i], f'filtered_image_sigma_{sigma}.png')
# Add Poisson noise
poisson_noisy_image = add_poisson_noise(original_image)
plt.subplot(2, 4, 8)
plt.imshow(poisson_noisy_image, cmap='gray')
cv2.imwrite("poison.png",poisson_noisy_image)
plt.title('Poisson Noisy Image')
plt.axis('off')


# # Add Salt and Pepper noise
# salt_pepper_noisy_image_1 = add_salt_and_pepper_noise(original_image, 0.01, 0.01)
# plt.subplot(2, 4, 2)
# plt.imshow(salt_pepper_noisy_image_1, cmap='gray')
# plt.title('S&P Noisy Image (0.01)')
# plt.axis('off')

# salt_pepper_noisy_image_2 = add_salt_and_pepper_noise(original_image, 0.02, 0.02)
# plt.subplot(2, 4, 3)
# plt.imshow(salt_pepper_noisy_image_2, cmap='gray')
# plt.title('S&P Noisy Image (0.02)')
# plt.axis('off')

cv2.imwrite("Saltpepperlenna.png",addition_noise(original_image,6000))
cv2.imshow('image-noise',original_image)
# Add Gaussian noise for different sigma values

# Adjust layout and display the plots
plt.tight_layout()
plt.show()