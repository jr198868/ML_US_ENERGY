from PIL import Image

import glob

# Specify the directory containing the PNG files
directory = '/Users/ranjing/Desktop/ML_US_ENERGY/ML_LCA'

# Get a list of all jpg files in the directory
jpg_files = glob.glob(directory + '/*.jpg')

# Iterate over each PNG file
for file_path in jpg_files:
    # Process each file as needed
    print('Processing file:', file_path)
    # Read the file using PIL or any other image library
    # For example, using PIL:
  

    
'''
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/Multiple Linear Regression Sensitivity Analysis.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/Support Vector Machine Sensitivity Analysis.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/K-Nearest Neighbors Sensitivity Analysis.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/Gradient Boosting Sensitivity Analysis.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/Decision Tree Actual CO₂ emissions vs Predicted CO₂ emissions.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/Decision Tree Sensitivity Analysis.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/Random Forest Sensitivity Analysis.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/Support Vector Machine Actual CO₂ emissions vs Predicted CO₂ emissions.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/Multiple Linear Regression Actual CO₂ emissions vs Predicted CO₂ emissions.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/K-Nearest Neighbors Actual CO₂ emissions vs Predicted CO₂ emissions.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/Gradient Boosting Actual CO₂ emissions vs Predicted CO₂ emissions.jpg
Processing file: /Users/ranjing/Desktop/ML_US_ENERGY/Random Forest Actual CO₂ emissions vs Predicted CO₂ emissions.jpg

'''   
    
    

# Load the six PNG images

image1 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/ML_LCA/Decision Tree Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
image2 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/ML_LCA/Random Forest Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
image3 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/ML_LCA/Multiple Linear Regression Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
image4 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/ML_LCA/K-Nearest Neighbors Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
image5 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/ML_LCA/Gradient Boosting Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
image6 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/ML_LCA/Support Vector Machine Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')

# Define the desired resolution
desired_width = 1920
desired_height = 1080

# Create a new blank canvas with the desired resolution
grouped_image = Image.new('RGB', (desired_width, desired_height), (255, 255, 255))

# Calculate the position to paste each image based on the desired grid layout
image_width = desired_width // 3  # Width of each image in the 3x2 grid
image_height = desired_height // 2  # Height of each image in the 3x2 grid

# Paste the images onto the grouped figure
grouped_image.paste(image1.resize((image_width, image_height), Image.ANTIALIAS), (0, 0))
grouped_image.paste(image2.resize((image_width, image_height), Image.ANTIALIAS), (image_width, 0))
grouped_image.paste(image3.resize((image_width, image_height), Image.ANTIALIAS), (2 * image_width, 0))
grouped_image.paste(image4.resize((image_width, image_height), Image.ANTIALIAS), (0, image_height))
grouped_image.paste(image5.resize((image_width, image_height), Image.ANTIALIAS), (image_width, image_height))
grouped_image.paste(image6.resize((image_width, image_height), Image.ANTIALIAS), (2 * image_width, image_height))



# Save the final grouped image
grouped_image.save('Actual CO₂ emissions vs CO₂ emissions based on LCA calculation of different machine learning methods.jpg')