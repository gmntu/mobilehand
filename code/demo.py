###############################################################################
### Demo MobileHand
### Input : 2D color image of hand
### Output: 3D hand shape and pose
###############################################################################

import cv2
import time
import torch
import argparse

from utils_display import Display
from utils_neural_network import HMR


# User selection
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--cuda', '-c', action='store_true', help='Use GPU')
parser.add_argument('--data', '-d', default='freihand',  help='stb / freihand')
parser.add_argument('--mode', '-m', default='image',     help='image / video / camera')
arg = parser.parse_args()

# Get device
device = torch.device('cuda' if torch.cuda.is_available() and arg.cuda else 'cpu')

# Load neural network
model = HMR(arg)
model.load_state_dict(torch.load('../model/hmr_model_' + arg.data + '_auc.pth'))
model.to(device)
model.eval()

# Load display
disp = Display(arg, model, device)

if arg.mode=='image':
    # Load image from file
    file = 'stb_SK_color_0.png' if arg.data=='stb' else 'freihand_00000000.jpg'
    img = cv2.imread('../data/' + file)

    # Special consideration for STB dataset
    if arg.data=='stb':
        img = cv2.flip(img, 1)                 # Mirror flip to convert left to right hand
        img = cv2.resize(img, (384,288))       # Reduce image size to 60% of [640,480] = [384,288] as simply cropping the hand is too big to fit into 224 by 224
        img = cv2.copyMakeBorder(img, 96, 96, 128, 128, cv2.BORDER_CONSTANT) # Add padding to maintain original size
        x,y = 267, 235                         # Location of index finger
        img = img[y-112:y+112, x-112:x+112, :] # Crop image to [224,224,3]

    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Convert to RGB
    img = torch.as_tensor(img, dtype=torch.float32) # Convert to torch tensor
    img = img.permute(2,0,1) / 255.0                # Shift RGB channel infront [3,224,224] and scale to [0,1]
    # Input image to model
    res = model(img.to(device).unsqueeze(0))
    # Display result
    disp.update(img, res)
    cv2.waitKey(60)
    disp.vis.run()

else:
    # Load video from .mp4 file or webcam (default index is usually 0 or 1)
    cap = cv2.VideoCapture('../data/video.mp4' if arg.mode=='video' else 0)

    while True:
        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Allow video to loop back
            ret, img = cap.read()

        # Start time
        t0 = time.time()
        # Preprocess image
        img = cv2.resize(img, (224, 224))               # Resize to [224,224,3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Convert to RGB
        img = torch.as_tensor(img, dtype=torch.float32) # Convert to torch tensor
        img = img.permute(2,0,1) / 255.0                # Shift RGB channel infront [3,224,224] and scale to [0,1]
        # Input image to model
        res = model(img.to(device).unsqueeze(0))
        # End time
        t1 = time.time() # Note: Only consider inference time and ignore time taken for display

        # Display result
        disp.update(img, res, t1 - t0)
        key = cv2.waitKey(1)
        if key==27: break # Press escape to end program

    cap.release()
    cv2.destroyAllWindows()