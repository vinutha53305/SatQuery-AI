from ai_model import SatelliteAI
from PIL import Image
import tkinter as tk
from tkinter import filedialog

print("1. Starting SatQuery AI VQA test...")

print("2. Loading AI model...")
ai = SatelliteAI()
print("3. AI model loaded successfully!")

print("4. Opening image selection window...")

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

image_path = filedialog.askopenfilename(
    parent=root,
    title="Select Satellite Image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not image_path:
    print("5. No image selected.")
    exit()

print("5. Selected image:")
print(image_path)

image = Image.open(image_path).convert("RGB")

question = input("\n6. Enter your question: ")

print("7. Sending image and question to AI...")

answer = ai.answer_question(
    image,
    question
)

print("\nQuestion:")
print(question)

print("\nAI Answer:")
print(answer)


