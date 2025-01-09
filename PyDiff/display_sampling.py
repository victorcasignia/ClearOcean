import os
from PIL import Image, ImageOps, ImageChops
import matplotlib.pyplot as plt

def display_images_horizontally(directory, space=10):
    # Get list of image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

    # Reverse the sorting of image files
    image_files.sort(reverse=True)
    
    # Load images
    images = [Image.open(os.path.join(directory, img_file)) for img_file in image_files]
    
    # Calculate total width and max height for the final image, including space between images
    total_width = sum(img.width for img in images) + space * (len(images) - 1)
    max_height = max(img.height for img in images)
    
    # Create a new blank image with the calculated dimensions and white background
    combined_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    # Paste images side by side with space between them
    x_offset = 0
    for img in images:
        # Create a white background image with the same height as the max height and width of the current image
        background = Image.new('RGB', (img.width, max_height), (255, 255, 255))
        # Calculate the vertical offset to center the image
        y_offset = (max_height - img.height) // 2
        # Paste the image onto the white background
        background.paste(img, (0, y_offset))
        # Paste the background onto the combined image
        combined_image.paste(background, (x_offset, 0))
        x_offset += img.width + space
    
    # Display the combined image
    plt.figure(figsize=(total_width // 100, 10))
    plt.imshow(combined_image)
    plt.axis('off')
    plt.show()


# Example usage
directory = '../../../Report/samples/preprocessing'
display_images_horizontally(directory)