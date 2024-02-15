import numpy as np
import matplotlib.pyplot as plt
import cv2

def resize_img(img,req_shape):
    
    #for upscaling, each pixel is duplicataed rfxcf times and appended appropriately in the resized matrix
    rf = req_shape[0] / img.shape[0]
    cf = req_shape[1] / img.shape[1]
    
    temp = []
    
    for i in range(img.shape[0]):
        Row = []
        for j in range(img.shape[1]):
            
            if len(img.shape) == 3:
                scl_pxl = [[img[i,j,:]]*int(cf)]*int(rf)
            
            else:
                scl_pxl = [[img[i,j]]*int(cf)]*int(rf)
            
            scl_pxl = np.array(scl_pxl)
            if j == 0:
                Row = scl_pxl
                #print("scaled pxl size: ",scl_pxl.shape)
            else:
                Row = np.concatenate((Row,scl_pxl), axis = 1)
              
        
        if i == 0:
            temp.append(Row)
            temp = np.array(Row)
            #print("rescaled Row of a row of org img size: ",Row.shape)
        
        else:
            temp = np.concatenate((temp,np.array(Row)))
    
    #print(temp.shape)
    return temp


def rgb_to_grey(img):
    temp = []
    p,q,r = (0.299,0.587,0.114)
    for row in range(len(img)):
        Row = []
        for column in range(len(img[0])):
            Row.append(img[row,column,0]*p + img[row,column,1]*q + img[row,column,2]*r)
        
        temp.append(Row)
    
    return np.array(temp)


def flip_image(img,mode):
    temp = []
    
    if mode == 'v':
        for i in range(img.shape[0] - 1 , -1 , -1):
            temp.append(img[i,:,:])
    
    elif mode == 'h':
        for i in range(img.shape[0]):
            Row = []
            for j in range(img.shape[1] - 1 , -1 , -1):
                Row.append(img[i,j,:])
            
            temp.append(Row)
    
    elif mode == 'c':
        for i in range(img.shape[0]):
            Row = []
            for j in range(img.shape[1]):
                cmap = []
                for k in range(img.shape[2]-1,-1,-1):
                    cmap.append(img[i,j,k])
                
                Row.append(cmap)
            
            temp.append(Row)
    
    temp = np.array(temp)
    
    #print("shape: ",temp.shape)
    
    return temp


def random_crop():    
    (rand_row,rand_col) = (np.random.randint(0,128) , np.random.randint(0,128))

    if rand_row < 128:
        if rand_col < 128:
            rand_crop = img_rgb_resize[rand_row:rand_row+128 , rand_col:rand_col+128 , :]
        else:
            rand_crop = img_rgb_resize[rand_row:rand_row+128 , rand_col-128:rand_col , :]
    else:
        if rand_col < 128:
            rand_crop = img_rgb_resize[rand_row-128:rand_row , rand_col:rand_col+128 , :]
        else:
            rand_crop = img_rgb_resize[rand_row-128:rand_row , rand_col-128:rand_col , :]

    rand_img = cv2.resize(rand_crop , (256,256) , interpolation = cv2.INTER_NEAREST)

    temp = rand_img[128,128,:]
    
    return temp


img = cv2.imread("test_11.JPEG")

img_rgb = flip_image(img , "c")

img_grey = rgb_to_grey(img_rgb)

######## Q1

img_rgb_resize = resize_img(img_rgb , (256,256))

img_grey_resize = resize_img(img_grey , (256,256))

##################################################################   Q2

# fig,(ax1,ax2) = plt.subplots(1,2)

# ax1.imshow(img_rgb_resize)
# ax2.imshow(img_grey_resize,cmap = 'gray')

# plt.show()

#################################################################  Q3

# cv2.imwrite("same_name.jpg",img_grey_resize)

# ###############################################################  Q4

# img_rgb_resize_vflip = flip_image(img_rgb_resize,'v')
# img_rgb_resize_hflip = flip_image(img_rgb_resize,'h')

# fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# fig.suptitle("Original image vs Vertical and Horizontal flip")


# ax1.imshow(img_rgb_resize)
# ax2.imshow(img_rgb_resize_vflip)

# ax1.set_title("Original")
# ax2.set_title("Vertical flipped")



# ax3.imshow(img_rgb_resize)
# ax4.imshow(img_rgb_resize_hflip)

# ax3.set_title("Original")
# ax4.set_title("Horizontal flipped")

# plt.show()

# #############################################################  Q5

centre_points = []

for i in range(128):
    
    Row = []
    
    for j in range(128):
        Row.append(random_crop())
    
    centre_points.append(Row)

centre_points = np.array(centre_points)

fig,(ax1,ax2) = plt.subplots(1,2)

ax1.imshow(img_rgb_resize)
ax2.imshow(centre_points)

plt.show()