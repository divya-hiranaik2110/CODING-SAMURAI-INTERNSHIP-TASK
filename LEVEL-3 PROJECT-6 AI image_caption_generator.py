import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration # type: ignore
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading

class ImageCaptionGenerator:
    def __init__(self):
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("AI Image Caption Generator")
        self.root.geometry("800x600")
        
        # Initialize the model and processor
        print("Loading AI model... Please wait...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Upload button
        self.upload_btn = ttk.Button(main_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, pady=10)
        
        # Image display
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.grid(row=1, column=0, pady=10)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()
        
        # Caption display
        self.caption_var = tk.StringVar()
        self.caption_var.set("Upload an image to generate caption")
        caption_label = ttk.Label(main_frame, textvariable=self.caption_var, wraplength=600)
        caption_label.grid(row=2, column=0, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, length=300, mode='indeterminate')
        self.progress.grid(row=3, column=0, pady=10)
        
        # History display
        history_frame = ttk.LabelFrame(main_frame, text="Caption History", padding="10")
        history_frame.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))
        
        self.history_text = tk.Text(history_frame, height=10, width=70)
        self.history_text.pack()
        
    def upload_image(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp")]
        )
        
        if file_path:
            # Start processing in a separate thread
            self.progress.start()
            self.upload_btn.configure(state='disabled')
            self.caption_var.set("Generating caption...")
            
            thread = threading.Thread(target=self.process_image, args=(file_path,))
            thread.daemon = True
            thread.start()
    
    def process_image(self, file_path):
        try:
            # Load and display image
            image = Image.open(file_path)
            # Resize image for display while maintaining aspect ratio
            display_size = (400, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Update UI in main thread
            self.root.after(0, self.update_image_display, photo)
            
            # Generate caption
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Update UI with caption
            self.root.after(0, self.update_caption, caption, file_path)
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
        finally:
            # Enable upload button and stop progress bar
            self.root.after(0, self.finish_processing)
    
    def update_image_display(self, photo):
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference
    
    def update_caption(self, caption, file_path):
        self.caption_var.set(caption)
        # Add to history
        filename = os.path.basename(file_path)
        history_entry = f"File: {filename}\nCaption: {caption}\n\n"
        self.history_text.insert(tk.END, history_entry)
        self.history_text.see(tk.END)
    
    def show_error(self, error_message):
        self.caption_var.set(f"Error: {error_message}")
    
    def finish_processing(self):
        self.progress.stop()
        self.upload_btn.configure(state='normal')
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageCaptionGenerator()
    app.run()

