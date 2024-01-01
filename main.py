import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from scipy.stats import entropy
import imghdr
# import matplotlib
# matplotlib.use('TkAgg')

# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import (
#     FigureCanvasTkAgg,
#     NavigationToolbar2Tk
# )

#import Image_processing

import pyautogui
scr_w, scr_h= pyautogui.size()

root = tk.Tk()
scr_w, scr_h = (1920, 1080)
root.geometry(f'{scr_w}x{scr_h}') 
root.title("Image Processing Application")
bg_color = '#f4e022'
root.configure(bg = bg_color)

# styles
OM_style = ttk.Style()
OM_style.configure('my.TMenubutton', font=('Arial', 16))

btn_style = ttk.Style()
btn_style.configure('my.TButton', font=('Arial', 16))

# operations
operation_btn_font = ('Arial', 14, 'bold')
operation_btn_pady = 5
operation_btn_padx = 5

#-----------------------------------------------------------------------------
# main label
main_lbl = tk.Label(root, text="Image Processing Application", bg = bg_color, foreground= '#de1b4a', font=('Arial', 30, 'bold'))
main_lbl.pack(pady=10)

# ************************
upper_part_bg = '#18224b'
image_square_bg = 'white'
# ************************
# ************************
label_width = 27
operation_frame_bg = '#5a37c3'
oper_txt_cl = '#220e24'
operation_btn_bg = '#2a3d66'
operation_btn_txt_cl = 'white'
# ************************

# images and righ side operations
upper_frame = Frame(root)
upper_frame.configure(bg = upper_part_bg)
upper_frame.pack(ipadx=10, ipady=5)

# images frame
img_frame = Frame(upper_frame)
img_frame.grid(row=0, column=0)
img_frame.configure(bg = upper_part_bg)

# image one label
image_one_label = tk.Label(img_frame, width= 50, height=20, bg=image_square_bg)
image_one_label.grid(padx=20, pady=20, row= 0, column=0)
 
# image one label
image_two_label = tk.Label(img_frame, width= 50, height=20, bg=image_square_bg)
image_two_label.grid(padx=20, pady=20, row=0, column=1)



def display_image_one(file_path):
    image = Image.open(file_path)
    image = image.resize((400, 300))
    photo = ImageTk.PhotoImage(image)
    image_one_label.config(image=photo, width=400, height=300)
    image_one_label.photo = photo
    
def open_image_one():
    file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
    if file_path:
        global selected_photo_path_1 
        selected_photo_path_1 = file_path
        img1 = cv2.imread(selected_photo_path_1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        global x, y
        x, y, z = img1.shape
        display_image_one(file_path) 
        
        
def display_image_two(file_path):
    image = Image.open(file_path)
    image = image.resize((400, 300))
    photo = ImageTk.PhotoImage(image)
    image_two_label.config(image=photo, width=400, height=300)
    image_two_label.photo = photo

def open_image_two():
    file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
    if file_path:
        global selected_photo_path_2
        selected_photo_path_2 = file_path
        display_image_two(file_path)

# //////////////////////////////// ---------------- functions ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def gray_image():
    if selected_photo_path_1:
        original_image = cv2.imread(selected_photo_path_1)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray_image, cmap="gray")
        plt.show()
        
def BGR():
    if selected_photo_path_1:
        original_image = cv2.imread(selected_photo_path_1)
        plt.imshow(original_image)
        plt.show()
        
def PIL_img():
    if selected_photo_path_1:
        im = Image.open(selected_photo_path_1)
        im.show()


def Image_Saturations(value):
    if selected_photo_path_1:
        img = Image.open(selected_photo_path_1)
        img_saturation = F.adjust_saturation(img, value)
        plt.imshow(img_saturation)
        plt.show()

def convert_to_binary():
    if selected_photo_path_1:
        original_image = cv2.imread(selected_photo_path_1,cv2.IMREAD_GRAYSCALE)
        _,binary=cv2.threshold(original_image,127,255,cv2.THRESH_BINARY)
        plt.imshow(binary,cmap='gray')
        plt.show()
    
def Arthimetic_Operations(value):
    if value == 'Add':
        if selected_photo_path_1:
            # Read the selected photo
            img1 = cv2.imread(selected_photo_path_1)
            img2 = cv2.imread(selected_photo_path_2)
            y, x, z = img2.shape
            img1 = cv2.resize(img1, (x, y))
            adding_img = cv2.add(img1, img2)
            plt.imshow(adding_img)
            plt.show()

    elif value == 'Sub' :
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1)
            img2 = cv2.imread(selected_photo_path_2)
            y, x, z = img2.shape
            img1 = cv2.resize(img1, (x, y))
            subtraction_img = cv2.subtract(img1, img2)
            plt.imshow(subtraction_img)
            plt.show()

    if value == 'Multi' :
        if selected_photo_path_1:
            # Read the selected photo
            img1 = cv2.imread(selected_photo_path_1)
            img2 = cv2.imread(selected_photo_path_2)
            y, x, z = img2.shape
            img1 = cv2.resize(img1, (x, y))
            multiplication = cv2.multiply(img1, img2)
            plt.imshow(multiplication)
            plt.show()

    elif value == 'Div' :
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1)
            img2 = cv2.imread(selected_photo_path_2)
            y, x, z = img2.shape
            img1 = cv2.resize(img1, (x, y))
            divide_img = cv2.divide(img1, img2)
            plt.imshow(divide_img)
            plt.show()

def logical_operations(value):
    if value == 'AND':
        if  selected_photo_path_1:
            # Read the selected photo
            img1 = cv2.imread( selected_photo_path_1)
            img2 = cv2.imread( selected_photo_path_2)
            y, x, z = img2.shape
            img1 = cv2.resize(img1, (x, y))
            and_img = cv2.bitwise_and(img1, img2)
            plt.imshow(and_img)
            plt.show()
    elif value == 'OR':
        if  selected_photo_path_1:
            img1 = cv2.imread( selected_photo_path_1)
            img2 = cv2.imread( selected_photo_path_2)
            y, x, z = img2.shape
            img1 = cv2.resize(img1, (x, y))
            Or_img = cv2.bitwise_or(img1, img2)
            plt.imshow(Or_img)
            plt.show()
    elif value == 'XOR':
        if  selected_photo_path_1:
            # Read the selected photo
            img1 = cv2.imread( selected_photo_path_1)
            img2 = cv2.imread( selected_photo_path_2)
            y, x, z = img2.shape
            img1 = cv2.resize(img1, (x, y))
            Xor_img = cv2.bitwise_xor(img1, img2)
            plt.imshow(Xor_img)
            plt.show()
    elif value == 'NOT_1':
        if  selected_photo_path_1:
            img1 = cv2.imread( selected_photo_path_1)
            Not_img = cv2.bitwise_not(img1)
            plt.imshow(Not_img)
            plt.show()
    elif value == 'NOT_2':
        if  selected_photo_path_2:
            img1 = cv2.imread( selected_photo_path_2)
            Not_img = cv2.bitwise_not(img1)
            plt.imshow(Not_img)
            plt.show()

def R_G_B():
    if selected_photo_path_1:
        img1 = cv2.imread(selected_photo_path_1)
        color_img = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        r_img = color_img.copy()
        r_img[:, :, (1, 2)] = 0
        g_img = color_img.copy()
        g_img[:, :, (0, 2)] = 0
        b_img = color_img.copy()
        b_img[:, :, (0, 1)] = 0
        plt.subplot(131), plt.imshow(r_img)
        plt.subplot(132), plt.imshow(g_img)
        plt.subplot(133), plt.imshow(b_img)
        plt.show()

def equalizeHist_img_one():
    if selected_photo_path_1:
        img1 = cv2.imread(selected_photo_path_1, cv2.IMREAD_GRAYSCALE)
        equalizeHist_img = cv2.equalizeHist(img1)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(equalizeHist_img, cmap='gray')
        plt.title('Equalized Image')

        plt.subplot(1, 2, 2)
        plt.hist(equalizeHist_img.ravel(), bins=256, range=[0, 256], color='r', alpha=0.5)
        plt.title('Equalized Histogram')
        plt.show()
        
def equalizeHist_img_two():
    if selected_photo_path_1:
        img1 = cv2.imread(selected_photo_path_2, cv2.IMREAD_GRAYSCALE)
        equalizeHist_img = cv2.equalizeHist(img1)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(equalizeHist_img, cmap='gray')
        plt.title('Equalized Image')

        plt.subplot(1, 2, 2)
        plt.hist(equalizeHist_img.ravel(), bins=256, range=[0, 256], color='r', alpha=0.5)
        plt.title('Equalized Histogram')
        plt.show()

def Thresholding_OTSU(thresh_type):
    if(thresh_type == "Binary"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1,cv2.IMREAD_GRAYSCALE)
            _, INV_image = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            plt.imshow(INV_image,cmap="gray")
            plt.show()
    elif (thresh_type == "Binary_INV"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1, cv2.IMREAD_GRAYSCALE)
            _, INV_image = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            plt.imshow(INV_image, cmap="gray")
            plt.show()
    elif (thresh_type == "Trunc"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1, cv2.IMREAD_GRAYSCALE)
            _, INV_image = cv2.threshold(img1, 120, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
            plt.imshow(INV_image, cmap="gray")
            plt.show()
    elif (thresh_type == "To_zero"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1, cv2.IMREAD_GRAYSCALE)
            _, INV_image = cv2.threshold(img1, 120, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
            plt.imshow(INV_image, cmap="gray")
            plt.show()
    elif (thresh_type == "To_zero_INV"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1, cv2.IMREAD_GRAYSCALE)
            _, INV_image = cv2.threshold(img1, 120, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
            plt.imshow(INV_image, cmap="gray")
            plt.show()
            
def Image_Rotation(centerX, centerY,angle):
    if selected_photo_path_1:
        img1 = cv2.imread(selected_photo_path_1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) 
        height, width = img1.shape[:2]
        M = cv2.getRotationMatrix2D((centerX, centerY), angle, 1.0)
        rotated = cv2.warpAffine(img1, M, (width, height))
        plt.imshow(rotated)
        plt.show()
        
def Gamma_Correction(gamma):
    if selected_photo_path_1:
        img1 = cv2.imread( selected_photo_path_1).astype(np.float32)/255
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        gamma_img = np.power(img, gamma)
        plt.imshow(gamma_img)
        plt.show()

def Filters(valu):
    if(valu == "Mean_Flilter"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            kernel = np.ones((5, 5), np.float32) / 25
            dst = cv2.filter2D(img1, -1, kernel)
            plt.imshow(dst)
            plt.show()
    elif (valu == "Gaussian_Filter"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            blur2 = cv2.GaussianBlur(img1, (5, 5), 0)
            plt.imshow(blur2)
            plt.show()
    elif (valu == "Median_Filter"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            median3 = cv2.medianBlur(img1, 5)
            plt.imshow(median3)
            plt.show()

    elif (valu == "Laplacian_Filter"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            i_sharpen = cv2.Laplacian(img1, cv2.CV_64F, 3)
            plt.imshow(i_sharpen)
            plt.show()
    elif (valu == "Unsharp_Filter"):
        if selected_photo_path_1:
            img1 = cv2.imread( selected_photo_path_1)
            im_blurred = cv2.GaussianBlur(img1, (11, 11), 10)
            im1 = cv2.addWeighted(img1, 1.0 + 3.0, im_blurred, -3.0, 0)  # im1 = im + 3.0*(im - im_blurred)
            plt.figure(figsize=(20, 10))
            plt.subplot(121), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.subplot(122), plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
            plt.show()

def Thresh(thresh_type):
    if(thresh_type == "Binary"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1,cv2.IMREAD_GRAYSCALE)
            _, INV_image = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
            plt.imshow(INV_image,cmap="gray")
            plt.show()
    elif (thresh_type == "Binary_INV"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1, cv2.IMREAD_GRAYSCALE)
            _, INV_image = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY_INV)
            plt.imshow(INV_image, cmap="gray")
            plt.show()
    elif (thresh_type == "Trunc"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1, cv2.IMREAD_GRAYSCALE)
            _, INV_image = cv2.threshold(img1, 120, 255, cv2.THRESH_TRUNC)
            plt.imshow(INV_image, cmap="gray")
            plt.show()
    elif (thresh_type == "To_zero"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1, cv2.IMREAD_GRAYSCALE)
            _, INV_image = cv2.threshold(img1, 120, 255, cv2.THRESH_TOZERO)
            plt.imshow(INV_image, cmap="gray")
            plt.show()
    elif (thresh_type == "To_zero_INV"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1, cv2.IMREAD_GRAYSCALE)
            _, INV_image = cv2.threshold(img1, 120, 255, cv2.THRESH_TOZERO_INV)
            plt.imshow(INV_image, cmap="gray")
            plt.show()
            
def resize(valu):
    if(valu == "CUBIC"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            resize_cubic = cv2.resize(img1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            plt.imshow(resize_cubic)
            plt.show()
    elif (valu == "AREA"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            resize_area = cv2.resize(img1, (0, 0), fx=0.5, fy=0.7, interpolation=cv2.INTER_AREA)
            plt.imshow(resize_area)
            plt.show()
    elif (valu == "LINEAR"):
        if selected_photo_path_1:
            img1 = cv2.imread(selected_photo_path_1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            resize_linear = cv2.resize(img1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            plt.imshow(resize_linear)
            plt.show()

def crop(st_x, fin_x, st_y, fin_y):
    img1 = cv2.imread(selected_photo_path_1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt.imshow(img1[st_x:fin_x, st_y:fin_y])
    plt.show()
    
def Adaptive(valu):
    if(valu == "MEAN_ADAPRIVVE"):
        if selected_photo_path_1:
            img = cv2.imread(selected_photo_path_1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 199, 5)
            plt.imshow(thresh1,cmap='gray')
            plt.show()
    elif (valu == "GAUSSIAN_ADAPTIVE"):
        if selected_photo_path_1:
            img = cv2.imread(selected_photo_path_1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 199, 5)
            plt.imshow(thresh2,cmap="gray")
            plt.show()
            
def histhistogram_img1():
    if selected_photo_path_1:
        img = cv2.imread(selected_photo_path_1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blue_channel = cv2.calcHist([img], [0], None, [256], [0, 256])
        red_channel = cv2.calcHist([img], [1], None, [256], [0, 256])
        green_channel = cv2.calcHist([img], [2], None, [256], [0, 256])
        plt.subplot(311)
        plt.bar(range(256), list(red_channel.flatten()), color='r')
        plt.subplot(312)
        plt.bar(range(256), list(blue_channel.flatten()),color='b')
        plt.subplot(313)
        plt.bar(range(256), list(green_channel.flatten()), color='g')
        plt.show()
        
def histhistogram_img2():
    if selected_photo_path_1:
        img = cv2.imread(selected_photo_path_2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blue_channel = cv2.calcHist([img], [0], None, [256], [0, 256])
        red_channel = cv2.calcHist([img], [1], None, [256], [0, 256])
        green_channel = cv2.calcHist([img], [2], None, [256], [0, 256])
        plt.subplot(311)
        plt.bar(range(256), list(red_channel.flatten()), color='r')
        plt.subplot(312)
        plt.bar(range(256), list(blue_channel.flatten()),color='b')
        plt.subplot(313)
        plt.bar(range(256), list(green_channel.flatten()), color='g')
        plt.show()
        
def blending(alph):
    if selected_photo_path_1:
        img1 = cv2.imread(selected_photo_path_1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(selected_photo_path_2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        y, x, z = img2.shape
        img1 = cv2.resize(img1, (x, y))
        blending = cv2.addWeighted(img1, alph, img2, (1-alph), 0)
        plt.imshow(blending)
        plt.show()

def reflection_hor(image):
    if image == 'Image One':
        image = cv2.imread(selected_photo_path_1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rows, clns, _ = image.shape
        mx = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
        img_reflected_x_axis = cv2.warpPerspective(image, mx, (clns, rows))
        plt.imshow(img_reflected_x_axis)
        plt.show()
        
    elif image == 'Image Two':
        image = cv2.imread(selected_photo_path_2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rows, clns, _ = image.shape
        mx = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
        img_reflected_x_axis = cv2.warpPerspective(image, mx, (clns, rows))
        plt.imshow(img_reflected_x_axis)
        plt.show()

def reflection_ver(image):
    if image == 'Image One':
        image = cv2.imread(selected_photo_path_1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rows, clns, _ = image.shape
        my = np.float32([[-1, 0, clns], [0, 1, 0], [0, 0, 1]])
        img_reflected_y_axis = cv2.warpPerspective(image, my, (clns, rows))
        plt.imshow(img_reflected_y_axis)
        plt.show()
        
    elif image == 'Image Two':
        image = cv2.imread(selected_photo_path_2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rows, clns, _ = image.shape
        my = np.float32([[-1, 0, clns], [0, 1, 0], [0, 0, 1]])
        img_reflected_y_axis = cv2.warpPerspective(image, my, (clns, rows))
        plt.imshow(img_reflected_y_axis)
        plt.show()
        
def translating(x, y):
   image_file = cv2.imread(selected_photo_path_1)
   image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
   T = np.float32([[1, 0, x], [0, 1, y]])
   width = image_file.shape[1]
   height = image_file.shape[0]
   shifted_img = cv2.warpAffine(image_file, T, (width, height))
   plt.imshow(shifted_img)
   plt.show()
   
def entropy_img1():
    img = cv2.imread(selected_photo_path_1, cv2.IMREAD_GRAYSCALE)
    hist, _ = np.histogram(img.ravel(), bins = 128, range  = (0, 128))
    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist, base = 2)
    plt.title(f'Imaeg Entropy: {image_entropy}')
    plt.imshow(img, cmap = 'gray')
    plt.show()

def entropy_img2():
    img = cv2.imread(selected_photo_path_2, cv2.IMREAD_GRAYSCALE)
    hist, _ = np.histogram(img.ravel(), bins = 128, range  = (0, 128))
    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist, base = 2)
    plt.title(f'Imaeg Entropy: {image_entropy}')
    plt.imshow(img, cmap = 'gray')
    plt.show()
    
def constrat_stretch_img2():
    image_object = Image.open(selected_photo_path_2)
    mutlibands = image_object.split()
    img = np.array(image_object)
    blue_channel = cv2.calcHist([img], [0], None, [256], [0, 256])
    red_channel = cv2.calcHist([img], [1], None, [256], [0, 256])
    green_channel = cv2.calcHist([img], [2], None, [256], [0, 256])
    plt.subplot(331)
    plt.bar(range(256), list(red_channel.flatten()), color='r')
    plt.subplot(332)
    plt.bar(range(256), list(blue_channel.flatten()),color='b')
    plt.subplot(333)
    plt.bar(range(256), list(green_channel.flatten()), color='g')
    # Normalized methods

    def Normalized_red(intensity):
      i_in = intensity
      max_out = 255
      min_out = 0
      max_in = 250
      min_in = 80
      i_out = (i_in - min_in) * (((max_out - min_out) / (max_in - min_in)) + min_out)
      return i_out
    
    def NormalizedGreen(intensity):
      i_in = intensity
      max_out = 255
      min_out = 0
      max_in = 220
      min_in = 100
      i_out = (i_in - min_in) * (((max_out - min_out) / (max_in - min_in)) + min_out)
      return i_out
    
    def NormalizedBlue(intensity):
      i_in = intensity
      max_out = 255
      min_out = 0
      max_in = 200
      min_in = 100
      i_out = (i_in - min_in) * (((max_out - min_out) / (max_in - min_in)) + min_out)
      return i_out
  
    normalizedRed = mutlibands[0].point(Normalized_red)
    normalizedGreen = mutlibands[1].point(NormalizedGreen)
    normalizedBlue = mutlibands[2].point(NormalizedBlue)
    normalizedImage = Image.merge("RGB", (normalizedRed, normalizedGreen, normalizedBlue))
    
    img_norm = np.array(normalizedImage)
    blue_channel = cv2.calcHist([img_norm], [0], None, [256], [0, 256])
    red_channel = cv2.calcHist([img_norm], [1], None, [256], [0, 256])
    green_channel = cv2.calcHist([img_norm], [2], None, [256], [0, 256])
    
    plt.subplot(334)
    plt.bar(range(256), list(red_channel.flatten()), color='r')
    plt.subplot(335)
    plt.bar(range(256), list(blue_channel.flatten()),color='b')
    plt.subplot(336)
    plt.bar(range(256), list(green_channel.flatten()), color='g')
    
    plt.subplot(337)
    plt.title("Before Constrast Stretching")
    plt.imshow(img)
    
    plt.subplot(339)
    plt.title("After Constrast Stretching")
    plt.imshow(img_norm)
    
    plt.tight_layout()
    plt.show()
    
def constrat_stretch_img1():
    image_object = Image.open(selected_photo_path_1)
    mutlibands = image_object.split()
    img = np.array(image_object)
    blue_channel = cv2.calcHist([img], [0], None, [256], [0, 256])
    red_channel = cv2.calcHist([img], [1], None, [256], [0, 256])
    green_channel = cv2.calcHist([img], [2], None, [256], [0, 256])
    plt.subplot(331)
    plt.bar(range(256), list(red_channel.flatten()), color='r')
    plt.subplot(332)
    plt.bar(range(256), list(blue_channel.flatten()),color='b')
    plt.subplot(333)
    plt.bar(range(256), list(green_channel.flatten()), color='g')
    # Normalized methods

    def Normalized_red(intensity):
      i_in = intensity
      max_out = 255
      min_out = 0
      max_in = 250
      min_in = 80
      i_out = (i_in - min_in) * (((max_out - min_out) / (max_in - min_in)) + min_out)
      return i_out
    
    def NormalizedGreen(intensity):
      i_in = intensity
      max_out = 255
      min_out = 0
      max_in = 220
      min_in = 100
      i_out = (i_in - min_in) * (((max_out - min_out) / (max_in - min_in)) + min_out)
      return i_out
    
    def NormalizedBlue(intensity):
      i_in = intensity
      max_out = 255
      min_out = 0
      max_in = 200
      min_in = 100
      i_out = (i_in - min_in) * (((max_out - min_out) / (max_in - min_in)) + min_out)
      return i_out
  
    normalizedRed = mutlibands[0].point(Normalized_red)
    normalizedGreen = mutlibands[1].point(NormalizedGreen)
    normalizedBlue = mutlibands[2].point(NormalizedBlue)
    normalizedImage = Image.merge("RGB", (normalizedRed, normalizedGreen, normalizedBlue))
    
    img_norm = np.array(normalizedImage)
    blue_channel = cv2.calcHist([img_norm], [0], None, [256], [0, 256])
    red_channel = cv2.calcHist([img_norm], [1], None, [256], [0, 256])
    green_channel = cv2.calcHist([img_norm], [2], None, [256], [0, 256])
    
    plt.subplot(334)
    plt.bar(range(256), list(red_channel.flatten()), color='r')
    plt.subplot(335)
    plt.bar(range(256), list(blue_channel.flatten()),color='b')
    plt.subplot(336)
    plt.bar(range(256), list(green_channel.flatten()), color='g')
    
    plt.subplot(337)
    plt.title("Before Constrast Stretching")
    plt.imshow(img)
    
    plt.subplot(339)
    plt.title("After Constrast Stretching")
    plt.imshow(img_norm)
    
    plt.tight_layout()
    plt.show()
    
    
def resolution(equality):
    im = Image.open(selected_photo_path_1)
    extension = imghdr.what(selected_photo_path_1)
    im.save('result_resolution.'+f'{extension}', quality = equality)
    img = cv2.imread('result_resolution.'+f'{extension}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    
    


   
# //////////////////////////////// ---------------- functions end ------------ \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
# Right side
select_button_img_one = tk.Button(img_frame, text="Select Image One", command=open_image_one)
select_button_img_one.configure(font=('Arial', 16, 'bold'), bg = operation_btn_bg, foreground=operation_btn_txt_cl)
select_button_img_one.grid(pady=20, row=1, column=0)

# Right side
select_button_img_two = tk.Button(img_frame, text="Select Image Two", command=open_image_two)
select_button_img_two.configure(font=('Arial', 16, 'bold'), bg = operation_btn_bg, foreground=operation_btn_txt_cl)
select_button_img_two.grid(pady=20, row=1, column=1)

#--------------------------------------------------------------------------------------
# right side operations
right_side_operations_frame = Frame(upper_frame)
right_side_operations_frame.grid(row=0, column=1)
right_side_operations_frame.configure(bg = upper_part_bg)

# Image Resize frame
image_resize_frame = Frame(right_side_operations_frame)
image_resize_frame.grid(row=0, column=0, padx=5, pady=5, ipadx=3, ipady=3)
image_resize_frame.configure(highlightthickness=2, highlightbackground='black', bg = operation_frame_bg)

# label
img_res_lbl = tk.Label(image_resize_frame, text="Image Resize", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg, width=label_width)
img_res_lbl.grid(row=0, column=0, padx=2, pady=2)

# interpolation combom box
interpolation_values = ['LINEAR', 'CUBIC', 'AREA']
interpolation_combbox = ttk.Combobox(image_resize_frame, values=interpolation_values, state="readonly", font=(('Arial', 14, 'bold')))
interpolation_combbox.grid(row=1, column=0, padx=2, pady=2)

# interpolation view btn
interpolation_btn = tk.Button(image_resize_frame, text="View", command= lambda: resize(interpolation_combbox.get()))
interpolation_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
interpolation_btn.grid(column=1, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# --------------------------------------------------------------------------------------
# Arithemtic operations
arithemtic_operation_frame = Frame(right_side_operations_frame)
arithemtic_operation_frame.grid(row=1, column=0, padx=5, pady=5, ipadx=3, ipady=3)
arithemtic_operation_frame.configure(highlightthickness=2, highlightbackground='black', bg = operation_frame_bg)

# label
arthi_oper_lbl = tk.Label(arithemtic_operation_frame, text="Arthimetic Operation", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg, width=label_width)
arthi_oper_lbl.grid(row=0, column=0, padx=2, pady=2)

# arithmeitic operations combom box
arthim_oper_values = ['Add', 'Sub', 'Multi', 'Div']
arthim_oper_combbox = ttk.Combobox(arithemtic_operation_frame, values=arthim_oper_values, state="readonly", font=(('Arial', 14, 'bold')))
arthim_oper_combbox.grid(row=1, column=0, padx=2, pady=2)

# arthimetic operations view btn
arthim_oper_btn = tk.Button(arithemtic_operation_frame, text="View", command= lambda : Arthimetic_Operations(arthim_oper_combbox.get()))
arthim_oper_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
arthim_oper_btn.grid(column=1, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# -----------------------------------------------------------------------------------------------------------
# Bitwise operations
bitwise_operation_frame = Frame(right_side_operations_frame)
bitwise_operation_frame.grid(row=2, column=0, padx=5, pady=5, ipadx=3, ipady=3)
bitwise_operation_frame.configure(highlightthickness=2, highlightbackground='black', bg = operation_frame_bg)

# label
bitwise_operation_lbl = tk.Label(bitwise_operation_frame, text="Bitwise Operation", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg, width=label_width)
bitwise_operation_lbl.grid(row=0, column=0, padx=2, pady=2)

# Bitwise operations combom box
bitwise_operation_values = ['AND', 'OR', 'XOR', 'NOT_1', 'NOT_2']
bitwise_operation_combbox = ttk.Combobox(bitwise_operation_frame, values=bitwise_operation_values, state="readonly", font=(('Arial', 14, 'bold')))
bitwise_operation_combbox.grid(row=1, column=0, padx=2, pady=2)

# Bitwise operations view btn
bitwise_operation_btn = tk.Button(bitwise_operation_frame, text="View", command= lambda: logical_operations(bitwise_operation_combbox.get()))
bitwise_operation_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
bitwise_operation_btn.grid(column=1, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

#------------------------------------------------------------------------------------------------------
# thresholding
thresholding_frame = Frame(right_side_operations_frame)
thresholding_frame.grid(row=0, column=1, padx=5, pady=5, ipadx=3, ipady=3)
thresholding_frame.configure(highlightthickness=2, highlightbackground='black', bg = operation_frame_bg)

# label
image_threshold_lbl = tk.Label(thresholding_frame, text="Image Thresholding", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg, width=label_width)
image_threshold_lbl.grid(row=0, column=0, padx=2, pady=2)

# thresholding options
thresholding_values = ['Binary', 'Binary_INV', 'Trunc', 'To_zero', 'To_zero_INV']
thresholding_comb = ttk.Combobox(thresholding_frame, values=thresholding_values, state="readonly", font=(('Arial', 14, 'bold')))
thresholding_comb.grid(row=1, column=0, padx=2, pady=2)

# thresholding view btn
thresholding_btn = tk.Button(thresholding_frame, text="View", command= lambda: Thresh(thresholding_comb.get()))
thresholding_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
thresholding_btn.grid(column=1, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# -------------------------------------------------------------------------------------------------
# adaptive thresholding
adaptive_thresholding_frame = Frame(right_side_operations_frame)
adaptive_thresholding_frame.grid(row=1, column=1, padx=5, pady=5, ipadx=3, ipady=3)
adaptive_thresholding_frame.configure(highlightthickness=2, highlightbackground='black', bg = operation_frame_bg)

# label
adaptive_threshold_lbl = tk.Label(adaptive_thresholding_frame, text="Image Adaptive Thresholding", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg, width=label_width)
adaptive_threshold_lbl.grid(row=0, column=0, padx=2, pady=2)

# adaptive thresholding options
adaptive_thresholding_values = ['MEAN_ADAPRIVVE', 'GAUSSIAN_ADAPTIVE']
adaptive_thresholding_comb = ttk.Combobox(adaptive_thresholding_frame, values=adaptive_thresholding_values, state="readonly", font=(('Arial', 14, 'bold')))
adaptive_thresholding_comb.grid(row=1, column=0, padx=2, pady=2)

# adaptive thresholding view btn
adaptive_thresholding_btn = tk.Button(adaptive_thresholding_frame, text="View", command=lambda: Adaptive(adaptive_thresholding_comb.get()))
adaptive_thresholding_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
adaptive_thresholding_btn.grid(column=1, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# ------------------------------------------------------------------------------------------------------
# OTSU thresholding
otsu_thresholding_frame = Frame(right_side_operations_frame)
otsu_thresholding_frame.grid(row=2, column=1, padx=5, pady=5, ipadx=3, ipady=3)
otsu_thresholding_frame.configure(highlightthickness=2, highlightbackground='black', bg = operation_frame_bg)

# label
image_threshold_lbl = tk.Label(otsu_thresholding_frame, text="Image OTSU Thresholding", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg, width=label_width)
image_threshold_lbl.grid(row=0, column=0, padx=2, pady=2)

# OTSU thresholding options
otsu_thresholding_values = ['Binary', 'Binary_INV', 'Trunc', 'To_zero', 'To_zero_INV']
otsu_thresholding_comb = ttk.Combobox(otsu_thresholding_frame, values=otsu_thresholding_values, state="readonly", font=(('Arial', 14, 'bold')))
otsu_thresholding_comb.grid(row=1, column=0, padx=2, pady=2)

# OTSU thresholding view btn
otsu_thresholding_btn = tk.Button(otsu_thresholding_frame, text="View", command= lambda: Thresholding_OTSU(otsu_thresholding_comb.get()))
otsu_thresholding_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
otsu_thresholding_btn.grid(column=1, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# --------------------------------------------------------------------------------------------------------
# filters
filers_frame = Frame(right_side_operations_frame)
filers_frame.grid(row=3, column=0, padx=5, pady=5, ipadx=3, ipady=3)
filers_frame.configure(highlightthickness=2, highlightbackground='black', bg = operation_frame_bg)

# label
filters_lbl = tk.Label(filers_frame, text="Image Filters", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg, width=label_width)
filters_lbl.grid(row=0, column=0, padx=2, pady=2)

# filters options
filers_values = ['Mean_Flilter', 'Gaussian_Filter', 'Median_Filter', 'Laplacian_Filter', 'Unsharp_Filter']
filers_comb = ttk.Combobox(filers_frame, values=filers_values, state="readonly", font=(('Arial', 14, 'bold')))
filers_comb.grid(row=1, column=0, padx=2, pady=2)

# filters view btn
filtes_btn = tk.Button(filers_frame, text="View", command= lambda: Filters(filers_comb.get()))
filtes_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
filtes_btn.grid(column=1, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# -----------------------------------------------------------------------------------

# bottom side
operations_frame = Frame(root)
operations_frame.configure(bg = bg_color)
operations_frame.pack(pady=10)

#----------------------------------------------------------------------------------------------
# image Crop
image_crop_frame = Frame(operations_frame)
image_crop_frame.grid(row=0, column=0, padx=5, pady=5, ipadx=10, ipady=3)
image_crop_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
img_crop_lbl = tk.Label(image_crop_frame, text="Image Crop", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg, width=10)
img_crop_lbl.grid(row=0, column=0, padx=2, pady=2)

# width label
crop_width_lbl = tk.Label(image_crop_frame, text="Width: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
crop_width_lbl.grid(row=1, column=0, padx=2, pady=2)

# from width label
crop_width_lbl_from = tk.Label(image_crop_frame, text="From: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
crop_width_lbl_from.grid(row=1, column=1, padx=2, pady=2)

# width start scale
start_crop_width_scale = tk.Entry(image_crop_frame, width=15)
start_crop_width_scale.grid(row=1, column=2, padx=2, pady=2)

# to width label
crop_width_lbl_to = tk.Label(image_crop_frame, text="To: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
crop_width_lbl_to.grid(row=1, column=3, padx=2, pady=2)

# width finish scale
fin_crop_width_scale = tk.Entry(image_crop_frame, width=15)
fin_crop_width_scale.grid(row=1, column=4, padx=2, pady=2)

#--------------------

# height label
crop_height_lbl = tk.Label(image_crop_frame, text="Height: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
crop_height_lbl.grid(row=2, column=0, padx=2, pady=2)

# from height label
crop_height_lbl_from = tk.Label(image_crop_frame, text="From: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
crop_height_lbl_from.grid(row=2, column=1, padx=2, pady=2)

# width start scale
start_crop_height_scale = tk.Entry(image_crop_frame, width=15)
start_crop_height_scale.grid(row=2, column=2, padx=2, pady=2)

# to width label
crop_height_lbl_to = tk.Label(image_crop_frame, text="To: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
crop_height_lbl_to.grid(row=2, column=3, padx=2, pady=2)

# width finish scale
fin_crop_height_scale = tk.Entry(image_crop_frame, width=15)
fin_crop_height_scale.grid(row=2, column=4, padx=2, pady=2)

# crop view btn
crop_view_btn = tk.Button(image_crop_frame, text="View", command=lambda: crop(int(float(start_crop_width_scale.get())), int(float(fin_crop_width_scale.get())), int(float(start_crop_height_scale.get())), int(float(fin_crop_height_scale.get()))))
crop_view_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
crop_view_btn.grid(column=0, row=3, padx=operation_btn_padx, pady= operation_btn_pady)

#-------------------------------------------------------------------------------------------
# Histrogram equalization
histogram_equal_frame = Frame(operations_frame)
histogram_equal_frame.grid(row=0, column=1, padx=5, pady=5, ipadx=3, ipady=18)
histogram_equal_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
histogram_equal_lbl = tk.Label(histogram_equal_frame, text="Histogram Equalization", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
histogram_equal_lbl.grid(row=0, column=0, padx=2, pady=2)

# histogram eqaulization view btn image one
histogram_equal_btn_img_one = tk.Button(histogram_equal_frame, text="Image One", command=equalizeHist_img_one)
histogram_equal_btn_img_one.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
histogram_equal_btn_img_one.grid(column=0, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# histogram eqaulization view btn
histogram_equal_btn_img_two = tk.Button(histogram_equal_frame, text="Image Two", command=equalizeHist_img_two)
histogram_equal_btn_img_two.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
histogram_equal_btn_img_two.grid(column=0, row=2, padx=operation_btn_padx, pady= operation_btn_pady)

# ----------------------------------------------------------------------------------------------------
# Histogram
histogram_frame = Frame(operations_frame)
histogram_frame.grid(row=0, column=3, padx=5, pady=5, ipadx=3, ipady=18)
histogram_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
histogram_lbl = tk.Label(histogram_frame, text="Image Histgram", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
histogram_lbl.grid(row=0, column=0, padx=2, pady=2)

# histogram image one btn
histogram_btn_img_one = tk.Button(histogram_frame, text="Imaeg One", command=histhistogram_img1)
histogram_btn_img_one.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
histogram_btn_img_one.grid(column=0, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# histogram image one btn
histogram_btn_img_two = tk.Button(histogram_frame, text="Image Two", command=histhistogram_img2)
histogram_btn_img_two.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
histogram_btn_img_two.grid(column=0, row=2, padx=operation_btn_padx, pady= operation_btn_pady)

# ----------------------------------------------------------------------------------------------
# image resolution frame
image_res_frame = Frame(operations_frame)
image_res_frame.grid(row=0, column=2, padx=5, pady=5, ipadx=3, ipady=18)
image_res_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
img_res_lbl = tk.Label(image_res_frame, text="Image Resolution", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
img_res_lbl.grid(row=0, column=0, padx=2, pady=2)

# scale
res_scale = Scale(image_res_frame, from_=0, to=100, orient=HORIZONTAL)
res_scale.configure(bg = operation_frame_bg, foreground=oper_txt_cl, highlightbackground=operation_frame_bg)
res_scale.grid(row=1, column=0, padx=operation_btn_padx, pady= operation_btn_pady)

# view btn
img_res_btn = tk.Button(image_res_frame, text="View", command= lambda: resolution(int(res_scale.get())))
img_res_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
img_res_btn.grid(column=0, row=2, padx=operation_btn_padx, pady= operation_btn_pady)

# --------------------------------------------------------------------------------------------------------------------
# constrast stretching
constrast_stretch_frame = Frame(operations_frame)
constrast_stretch_frame.grid(row=0, column=4, padx=5, pady=5, ipadx=3, ipady=18)
constrast_stretch_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
constrast_stretch_lbl = tk.Label(constrast_stretch_frame, text="Constrat Stretching", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
constrast_stretch_lbl.grid(row=0, column=0, padx=2, pady=2)

# constrast stretching view btn image one
constrast_str_btn_image_one = tk.Button(constrast_stretch_frame, text="Image One", command= constrat_stretch_img1)
constrast_str_btn_image_one.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
constrast_str_btn_image_one.grid(column=0, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# constrast stretching view btn image two
constrast_str_btn_image_two = tk.Button(constrast_stretch_frame, text="Image Two", command= constrat_stretch_img2)
constrast_str_btn_image_two.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
constrast_str_btn_image_two.grid(column=0, row=2, padx=operation_btn_padx, pady= operation_btn_pady)

# --------------------------------------------------------------------------------------------------
# image translation
image_translating_frame = Frame(operations_frame)
image_translating_frame.grid(row=0, column=5, padx=5, pady=5, ipadx=3, ipady=13)
image_translating_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
image_translating_lbl = tk.Label(image_translating_frame, text="Image Tanslating", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
image_translating_lbl.grid(row=0, column=0, padx=2, pady=2)

# x label
x_translating_lbl = tk.Label(image_translating_frame, text="X Value: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg,)
x_translating_lbl.grid(row=1, column=0, padx=2, pady=2)

# x entry
x_translating_entry = Entry(image_translating_frame)
x_translating_entry.grid(row=1, column=1)

# y label
y_translating_lbl = tk.Label(image_translating_frame, text="Y Value: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg,)
y_translating_lbl.grid(row=2, column=0, padx=2, pady=2)

# y entry
y_translating_entry = Entry(image_translating_frame)
y_translating_entry.grid(row=2, column=1)

# image translating view btn
image_translating_btn = tk.Button(image_translating_frame, text="View", command= lambda: translating(float(x_translating_entry.get()), float(y_translating_entry.get())))
image_translating_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
image_translating_btn.grid(column=0, row=3, padx=operation_btn_padx, pady= operation_btn_pady)

# --------------------------------------------------------------------------------------------------------
# Image Entropy
image_entropy_frame = Frame(operations_frame)
image_entropy_frame.grid(row=0, column=6, padx=5, pady=5, ipadx=3, ipady=18)
image_entropy_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
image_entropy_lbl = tk.Label(image_entropy_frame, text="Image Entropy", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
image_entropy_lbl.grid(row=0, column=0, padx=2, pady=2)

# image entropy view btn image one
image_entropy_btn_img_one = tk.Button(image_entropy_frame, text="Image One", command= entropy_img1)
image_entropy_btn_img_one.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
image_entropy_btn_img_one.grid(column=0, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# image entropy view btn
image_entropy_btn_img_two = tk.Button(image_entropy_frame, text="Image Two", command= entropy_img2)
image_entropy_btn_img_two.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
image_entropy_btn_img_two.grid(column=0, row=2, padx=operation_btn_padx, pady= operation_btn_pady)

#-------------------------------- row zero finish ----------------------------------------------
row_one_frame = Frame(root)
row_one_frame.configure(bg = bg_color)
row_one_frame.pack()
# ----------------------------------------------------------------------------------------------

# image saturation
image_saturation_frame = Frame(row_one_frame)
image_saturation_frame.grid(row=0, column=0, padx=5, pady=5, ipadx=3, ipady=30)
image_saturation_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
img_res_lbl = tk.Label(image_saturation_frame, text="Image Saturation", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
img_res_lbl.grid(row=0, column=0, padx=2, pady=2)

# Saturation entry
saturation_entry = Entry(image_saturation_frame)
saturation_entry.grid(row=1, column=0)

# saturation view btn
saturation_view_btn = tk.Button(image_saturation_frame, text="View", command=lambda: Image_Saturations(float(saturation_entry.get())))
saturation_view_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
saturation_view_btn.grid(column=0, row=2, padx=operation_btn_padx, pady= operation_btn_pady)

# -------------------------------------------------------------------------------------------------------
# image rotation
image_rotation_frame = Frame(row_one_frame)
image_rotation_frame.grid(row=0, column=1, padx=5, pady=5, ipadx=3, ipady=13)
image_rotation_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
image_rotation_lbl = tk.Label(image_rotation_frame, text="Image Rotation", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
image_rotation_lbl.grid(row=0, column=0, padx=2, pady=2) 

# center lbl
center_rotation_lbl = tk.Label(image_rotation_frame, text="Center: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
center_rotation_lbl.grid(row=1, column=0, padx=2, pady=2)

# x label
x_rotation_lbl = tk.Label(image_rotation_frame, text="X Value: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
x_rotation_lbl.grid(row=1, column=1, padx=2, pady=2)

# x entry
x_rotation_entry = Entry(image_rotation_frame)
x_rotation_entry.grid(row=1, column=2)

# y label
y_rotation_lbl = tk.Label(image_rotation_frame, text="Y Value: ", font=('Arial', 12, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
y_rotation_lbl.grid(row=2, column=1, padx=2, pady=2)

# y entry
y_rotation_entry = Entry(image_rotation_frame)
y_rotation_entry.grid(row=2, column=2)

# angle scale
angle_rotation_scale = Scale(image_rotation_frame, from_=0, to=360, orient=HORIZONTAL)
angle_rotation_scale.configure(bg = operation_frame_bg, foreground=oper_txt_cl, highlightbackground=operation_frame_bg)
angle_rotation_scale.grid(row=3, column=0)

# rotation view btn
image_rotation_btn = tk.Button(image_rotation_frame, text="View", command=lambda: Image_Rotation(float(x_rotation_entry.get()), float(y_rotation_entry.get()), int(angle_rotation_scale.get())))
image_rotation_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
image_rotation_btn.grid(column=2, row=3, padx=operation_btn_padx, pady= operation_btn_pady)

# -------------------------------------------------------------------------------------------
# Blending
blending_frame = Frame(row_one_frame)
blending_frame.grid(row=0, column=2, padx=5, pady=5, ipadx=3, ipady=30)
blending_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
blending_lbl = tk.Label(blending_frame, text="Blending Images", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
blending_lbl.grid(row=0, column=0, padx=2, pady=2)

# alpha entry
alph_entry = Scale(blending_frame, from_=0, to=1, orient=HORIZONTAL, resolution=0.1)
alph_entry.configure(bg = operation_frame_bg, foreground=oper_txt_cl, highlightbackground=operation_frame_bg)
alph_entry.grid(row=1, column=0)

# blending operations view btn
blending_btn = tk.Button(blending_frame, text="View", command= lambda: blending(float(alph_entry.get())))
blending_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
blending_btn.grid(column=0, row=2, padx=operation_btn_padx, pady= operation_btn_pady)

# ----------------------------------------------------------------------------------------------------
# image reflection
image_reflection_frame = Frame(row_one_frame)
image_reflection_frame.grid(row=0, column=3, padx=5, pady=5, ipadx=3, ipady=3)
image_reflection_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
image_translating_lbl = tk.Label(image_reflection_frame, text="Image Reflection", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
image_translating_lbl.grid(row=0, column=0, padx=2, pady=2)

# image reflection options
image_reflection_values = ['Image One', 'Image Two']
image_reflection_comb = ttk.Combobox(image_reflection_frame, values=image_reflection_values, state="readonly", font=(('Arial', 14, 'bold')))
image_reflection_comb.grid(row=1, column=0, padx=2, pady=2)

# horizontal btn
image_reflection_btn_img_hor = tk.Button(image_reflection_frame, text="Horizontal", command= lambda: reflection_hor(image_reflection_comb.get()))
image_reflection_btn_img_hor.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
image_reflection_btn_img_hor.grid(column=0, row=2, padx=operation_btn_padx, pady= operation_btn_pady)

# vertical btn
image_reflection_btn_img_ver = tk.Button(image_reflection_frame, text="Vertical", command= lambda: reflection_ver(image_reflection_comb.get()))
image_reflection_btn_img_ver.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
image_reflection_btn_img_ver.grid(column=0, row=3, padx=operation_btn_padx, pady= operation_btn_pady)

# -----------------------------------------------------------------------------------------
# Gamma Correction
gamma_correction_frame = Frame(row_one_frame)
gamma_correction_frame.grid(row=0, column=4, padx=5, pady=5, ipadx=3, ipady=41)
gamma_correction_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
gamma_correction_lbl = tk.Label(gamma_correction_frame, text="Gamma Correction", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
gamma_correction_lbl.grid(row=0, column=0, padx=2, pady=2)

# gamma entry
gamma_entry = Entry(gamma_correction_frame)
gamma_entry.grid(row=1, column=0)

# gamma view btn
gamma_view_btn = tk.Button(gamma_correction_frame, text="View", command=lambda: Gamma_Correction(float(gamma_entry.get())))
gamma_view_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
gamma_view_btn.grid(column=1, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# ------------------------------------------------------------------------------------------------------
# color conversion frame
color_cv_frame = Frame(root)
color_cv_frame.pack(padx=5, pady=20, ipadx=30, ipady=15)
color_cv_frame.configure(highlightthickness=2, highlightbackground='black', bg=operation_frame_bg)

# label
color_cv_lbl = tk.Label(color_cv_frame, text="Color Conversion", font=('Arial', 14, 'bold'), foreground = oper_txt_cl , bg = operation_frame_bg)
color_cv_lbl.grid(row=0, column=0, padx=2, pady=2)

# RGB to Gray
convert_gray_btn = tk.Button(color_cv_frame, text="Gray Image", command= gray_image)
convert_gray_btn.configure(font = operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
convert_gray_btn.grid(column=0, row=1, padx=operation_btn_padx, pady= operation_btn_pady)
        
# Gray to Binary
convert_bin_btn = tk.Button(color_cv_frame, text="Binary Image", command=convert_to_binary)
convert_bin_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
convert_bin_btn.grid(column=1, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# RGB to BGR
convert_BGR_btn = tk.Button(color_cv_frame, text="BGR", command= BGR)
convert_BGR_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
convert_BGR_btn.grid(column=2, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# RGB
convert_RGB_btn = tk.Button(color_cv_frame, text="RGB", command=R_G_B)
convert_RGB_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
convert_RGB_btn.grid(column=3, row=1, padx=operation_btn_padx, pady= operation_btn_pady)

# local
local_view_btn = tk.Button(color_cv_frame, text="View In Local Viewer", command=PIL_img)
local_view_btn.configure(font=operation_btn_font, bg = operation_btn_bg, foreground=operation_btn_txt_cl)
local_view_btn.grid(column=4, row=1, padx=operation_btn_padx, pady= operation_btn_pady)


# mainLoop
root.mainloop()
import os
extension = imghdr.what(selected_photo_path_1)
os.remove('result_resolution.'+f'{extension}')

