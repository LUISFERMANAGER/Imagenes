import cv2

# Load the pre-trained Haar Cascade classifier for cat faces
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

def detect_cat(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect cats
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected cats
    for (x, y, w, h) in cats:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the output
    cv2.imshow('Cat Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_cat('static/domesticos.jpg')