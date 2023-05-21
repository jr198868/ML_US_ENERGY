from PIL import Image

import glob

# Specify the directory containing the PNG files
directory = '/Users/ranjing/Desktop/ML_US_ENERGY'

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
'''
image1 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Decision Tree Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
image2 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Random Forest Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
image3 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Multiple Linear Regression Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
image4 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/K-Nearest Neighbors Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
image5 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Gradient Boosting Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
image6 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Support Vector Machine Actual CO₂ emissions vs Predicted CO₂ emissions.jpg')
'''

image1 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Decision Tree Sensitivity Analysis.jpg')
image2 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Random Forest Sensitivity Analysis.jpg')
image3 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Multiple Linear Regression Sensitivity Analysis.jpg')
image4 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Multiple Linear Regression Sensitivity Analysis.jpg')
image5 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Gradient Boosting Sensitivity Analysis.jpg')
image6 = Image.open('/Users/ranjing/Desktop/ML_US_ENERGY/Support Vector Machine Sensitivity Analysis.jpg')

# Define the desired resolution
desired_width = 1920
desired_height = 1080

# Resize the images to a consistent size (assuming all images have the same dimensions)
width, height = image1.size
resized_width = width // 2
resized_height = height // 3
image1 = image1.resize((resized_width, resized_height))
image2 = image2.resize((resized_width, resized_height))
image3 = image3.resize((resized_width, resized_height))
image4 = image4.resize((resized_width, resized_height))
image5 = image5.resize((resized_width, resized_height))
image6 = image6.resize((resized_width, resized_height))


max_width = max(image1.width, image2.width, image3.width, image4.width, image5.width, image6.width)
max_height = max(image1.height, image2.height, image3.height, image4.height, image5.height, image6.height)

grouped_width = max_width * 2
grouped_height = max_height * 3
grouped_image = Image.new('RGB', (grouped_width, grouped_height), (255, 255, 255))

# Paste the images onto the grouped figure
grouped_image.paste(image1, (0, 0))
grouped_image.paste(image2, (max_width, 0))
grouped_image.paste(image3, (0, max_height))
grouped_image.paste(image4, (max_width, max_height))
grouped_image.paste(image5, (0, 2 * max_height))
grouped_image.paste(image6, (max_width, 2 * max_height))

# Save the final grouped image
#grouped_image.save('Actual CO₂ emissions vs Predicted CO₂ emissions of different machine learning methods.jpg')
grouped_image.save('Sensitivity Analysis of different machine learning methods: Changes in the R² when the selected variable is perturbed.jpg')