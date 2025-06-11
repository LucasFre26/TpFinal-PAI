import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from predict import predict_pneumonia
import threading
import time

def upload_images():
    file_paths = filedialog.askopenfilenames()
    if file_paths:
        progress['maximum'] = len(file_paths)
        for idx, file_path in enumerate(file_paths):
            img = Image.open(file_path).resize((224, 224))
            img_tk = ImageTk.PhotoImage(img)
            panel.configure(image=img_tk)
            panel.image = img_tk
            result, confidence = predict_pneumonia(file_path)
            label_result.config(text=f'Result: {result} ({confidence:.2f}%)')
            progress['value'] = idx + 1
            root.update_idletasks()
            time.sleep(0.5)  # Simula tempo de processamento

# GUI setup
root = tk.Tk()
root.title('Detector de Pneumonia')
root.configure(bg='#2e2e2e')

# Title
title = tk.Label(root, text='Detector de Pneumonia', font=('Arial', 20, 'bold'),
                 bg='#2e2e2e', fg='white')
title.pack(pady=10)

# Image panel
panel = tk.Label(root, bg='#2e2e2e')
panel.pack(pady=10)

# Progress bar
progress = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
progress.pack(pady=10)

# Result
label_result = tk.Label(root, text='Resultado:', font=('Arial', 14),
                        bg='#2e2e2e', fg='white')
label_result.pack(pady=10)

# Upload button
btn_upload = tk.Button(root, text='Upload Image(s)', command=lambda: threading.Thread(target=upload_images).start(),
                       bg='#444', fg='white', font=('Arial', 12, 'bold'))
btn_upload.pack(pady=10)

# Animation
def animate():
    while True:
        for frame in ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]:
            anim_label.config(text=frame)
            time.sleep(0.1)

anim_label = tk.Label(root, font=('Arial', 20), bg='#2e2e2e', fg='white')
anim_label.pack()

threading.Thread(target=animate, daemon=True).start()

root.mainloop()
