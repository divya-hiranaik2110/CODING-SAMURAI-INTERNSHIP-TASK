import cv2
import os

# Define expression categories
expressions = ['happy', 'sad', 'angry', 'neutral', 'surprise']
dataset_path = "dataset"

# Ensure dataset structure exists
for exp in expressions:
    os.makedirs(os.path.join(dataset_path, exp), exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

# User chooses an expression category
expression = input("Enter expression name (happy/sad/angry/neutral/surprise): ").lower()
if expression not in expressions:
    print("Invalid expression! Choose from:", expressions)
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Define path to save images
save_path = os.path.join(dataset_path, expression)
count = 0

print(f"📸 Capturing images for '{expression}' expression. Press 'q' to quit.")

while count < 50:  # Capture 50 images
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not access webcam.")
        break

    # Display the frame
    cv2.imshow("Capture", frame)

    # Save the captured image
    file_name = os.path.join(save_path, f"{count}.jpg")
    cv2.imwrite(file_name, frame)
    print(f"Saved: {file_name}")
    
    count += 1

    # Press 'q' to quit early
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"✅ Captured {count} images for '{expression}' category.")
