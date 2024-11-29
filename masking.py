import tensorflow as tf

def create_mask(pred_mask, n_h_ori, n_w_ori):
    # Get the class with the highest score for each pixel
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = pred_mask[0]
    pred_mask = tf.where((pred_mask == 0) | (pred_mask == 2), 255, 0)
    pred_mask= tf.image.resize(pred_mask, [n_h_ori, n_w_ori], method = "nearest")
    
    pred_mask = tf.squeeze(pred_mask, axis=-1)

    return pred_mask

# def creat_masked_images(pred_mask):
#     pred_mask = create_mask(pred_mask)
#     # Convert tensors to NumPy arrays for OpenCV processing
#     pred_mask_np = pred_mask.numpy().astype(np.uint8)  # Convert predicted mask to NumPy
#     # Preprocess the predicted mask: 
#     # Make the object (label 0) and border (label 2) white (255) and the background (label 1) black (0)
#     pred_mask_np = (np.where((pred_mask_np == 0) | (pred_mask_np == 2), 255, 0)).astype(np.uint8)
    
#     return pred_mask_np
