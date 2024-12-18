import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from masking import create_mask
# import os

n_h, n_w = 128, 128

# Parse function for input image
def parse_function(file_image_path):
    image_string = tf.io.read_file(file_image_path)
    image = tf.image.decode_image(image_string, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Scale to [0, 1]
    n_h_ori, n_w_ori, _ = image.shape
    image = tf.image.resize(image, [n_h, n_w])  # Resize to target dimensions
    return image, n_h_ori, n_w_ori

# Main processing pipeline
def pipeline_wrapper(image_path):
    unet_model = load_model("unetV1.keras")
    image, n_h_ori, n_w_ori = parse_function(image_path)
    pred_mask = create_mask(unet_model.predict(image[tf.newaxis,]), n_h_ori, n_w_ori)
    return pred_mask.numpy()

# Get all image paths in the same folder
# image_folder = 'ex_images'  # Directory of the script
# image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_paths = ['ex_images/cat_ex2.jpeg','ex_images/cat_ex3jpg.jpg','ex_images/dog_ex1.png','ex_images/dog_ex2.jpg']

# Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Input Image")
            gr.Markdown("**Supported Classes:** Cats, Dogs")
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Processed Mask")
            start_btn = gr.Button(value="Masking")
    
    # Link button to the wrapper function
    start_btn.click(pipeline_wrapper, inputs=input_image, outputs=output_image)

    # Add examples
    examples = gr.Examples(examples=image_paths, inputs=input_image)

# Launch the app
demo.launch(share=True)
