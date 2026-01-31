import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import nibabel as nib
import numpy as np
from PIL import Image
import torchvision.transforms as transforms  # Make sure this is imported

def visualize_gradcam(model, df, class_names, img_dir, device, num_images=5, target_class_idx=0, manual_bboxes=None):
    model.eval()
    selected_samples = df.sample(n=num_images)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    target_layers = [model.features[-1]]  # Last DenseNet conv layer

    for i, (idx, row) in enumerate(selected_samples.iterrows()):
        image_path = row['filepath']
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        try:
            img_data = nib.load(image_path).get_fdata()
            mid_slice = img_data.shape[2] // 2
            image_np = img_data[:, :, mid_slice]
            image = Image.fromarray(image_np).convert("RGB")
        except FileNotFoundError:
            print(f"Image not found at path: {image_path}")
            continue
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        original_image = np.array(image.resize((224, 224))) / 255.0
        input_tensor = transform(image).unsqueeze(0).to(device)

        targets = [ClassifierOutputTarget(target_class_idx)]

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        visualization = show_cam_on_image(original_image.astype(np.float32), grayscale_cam, use_rgb=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(visualization)
        ax.set_title(f"Grad-CAM for {class_names[target_class_idx]}")
        ax.axis('off')

        if manual_bboxes and i < len(manual_bboxes) and manual_bboxes[i] is not None:
            x_min, y_min, x_max, y_max = manual_bboxes[i]
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        plt.show()

    return grayscale_cam

# draw manual boxes if needed
manual_boxes = [None,None, None, None]

visualize_gradcam(model, df_val, class_names, train_root, device, target_class_idx=9, manual_bboxes=manual_boxes)


