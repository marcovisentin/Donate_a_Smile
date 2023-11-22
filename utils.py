import cv2
 
def detect_smile(gray_frame, faces, cascade_smile):
    smiles = []
    for (x, y, w, h) in faces:
        the_face = gray_frame[y:y+h, x:x+w] # get face bounding box
        smiles = cascade_smile.detectMultiScale(the_face,scaleFactor=2, minNeighbors=35, minSize = (32,64)) # detect smile
        for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0,255,0), 2) 
    return len(smiles) > 0

def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    """
     Overlay a PNG image with transparency onto another image using alpha blending.
     The function handles out-of-bound positions, including negative coordinates, by cropping
     the overlay image accordingly. Edges are smoothed using alpha blending.

     :param imgBack: The background image, a NumPy array of shape (height, width, 3) or (height, width, 4).
     :param imgFront: The foreground PNG image to overlay, a NumPy array of shape (height, width, 4).
     :param pos: A list specifying the x and y coordinates (in pixels) at which to overlay the image.
                 Can be negative or cause the overlay image to go out-of-bounds.
     :return: A new image with the overlay applied, a NumPy array of shape like `imgBack`.
     """
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape

    x1, y1 = max(pos[0], 0), max(pos[1], 0)
    x2, y2 = min(pos[0] + wf, wb), min(pos[1] + hf, hb)

    # For negative positions, change the starting position in the overlay image
    x1_overlay = 0 if pos[0] >= 0 else -pos[0]
    y1_overlay = 0 if pos[1] >= 0 else -pos[1]

    # Calculate the dimensions of the slice to overlay
    wf, hf = x2 - x1, y2 - y1

    # If overlay is completely outside background, return original background
    if wf <= 0 or hf <= 0:
        return imgBack

    # Extract the alpha channel from the foreground and create the inverse mask
    alpha = imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 3] / 255.0
    inv_alpha = 1.0 - alpha

    # Extract the RGB channels from the foreground
    imgRGB = imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 0:3]

    # Alpha blend the foreground and background
    for c in range(0, 3):
        imgBack[y1:y2, x1:x2, c] = imgBack[y1:y2, x1:x2, c] * inv_alpha + imgRGB[:, :, c] * alpha

    return imgBack