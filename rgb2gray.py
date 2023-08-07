import cv2
import glob

# Specify the path to the folder containing the JPEG images
image_folder = "data/nerf_llff_data/RedCar/images/*.jpeg"
output_folder = "RedCar_gray/"

# Load and convert images to grayscale
for image_file in glob.glob(image_folder):
    print(image_file)
    image = cv2.imread(image_file)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract the filename from the image path
    filename = image_file.split("/")[-1]
    
    # Save the grayscale image
    output_path = output_folder + filename
    cv2.imwrite(output_path, grayscale_image)

print("Grayscale images saved successfully!")
