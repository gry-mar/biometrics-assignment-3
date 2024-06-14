import tkinter as tk
from tkinter import filedialog, Label, Entry, messagebox
import shutil
import os

class AddUser:
    def __init__(self, master):
        self.master = master
        master.title("Add user")
        master.geometry("600x400")

        Label(master, text="Name:").pack()
        self.name_entry = Entry(master, width=50)
        self.name_entry.pack()

        Label(master, text="Surname:").pack()
        self.surname_entry = Entry(master, width=50)
        self.surname_entry.pack()

        self.drop_area = Label(master, text="Click here to select images", width=60, height=10, bg="lightgrey")
        self.drop_area.pack(pady=20)
        self.drop_area.bind("<Button-1>", self.select_images)

        self.save_button = tk.Button(master, text="Save Images", command=self.save_images, state="disabled")
        self.save_button.pack(pady=20)

        self.selected_image_paths = []

    def select_images(self, event):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_paths:
            self.selected_image_paths = file_paths
            file_list = ", ".join([os.path.basename(path) for path in file_paths])
            self.drop_area.config(text=f"Selected: {file_list}")
            self.save_button["state"] = "normal"

    def save_images(self):
        name = self.name_entry.get().strip()
        surname = self.surname_entry.get().strip()
        if not name or not surname:
            messagebox.showerror("Error", "Please enter both name and surname.")
            return

        new_location = "../dataset/dataset_images"
        if not os.path.exists(new_location):
            os.makedirs(new_location)

        for i, path in enumerate(self.selected_image_paths):
            new_name = f"{name}_{surname}_profile_{i}{os.path.splitext(path)[-1]}"
            new_path = os.path.join(new_location, new_name)
            shutil.copy(path, new_path)

        messagebox.showinfo("Success", "Images saved successfully.")
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AddUser(root)
    root.mainloop()
