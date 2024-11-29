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
