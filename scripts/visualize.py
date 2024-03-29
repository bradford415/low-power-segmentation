"""Script to add a legend to a segmented image (rgb, not grayscale)"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image


def label_segmented_image(segmented_images, color_palette):
    """Label random segmented image and add class labels

    Args:
        segmented_images: List of file paths to the segmented images
        color_palette: Dictionary that maps the label or color to the object class
    """
    index = np.random.choice(len(segmented_images))
    image = mpimg.imread(segmented_images[index])
    print(f'Loading random segmented image: {Path(segmented_images[index]).stem}')
    # Convert color palette to list of colors and labels
    colors = [[value['color'][0] / 255, value['color'][1] / 255, value['color'][2] / 255]
               for key, value in color_palette.items()]
    labels = [value['name'] for key, value in color_palette.items()]

    plt.figure(figsize=(9, 9))
    ax = plt.imshow(image)
    plt.title(Path(segmented_images[index]).stem)
    plt.axis('off')
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(
        l=labels[i])) for i in range(len(labels))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, ncol=2,
               handleheight=2, handlelength=1.5, labelspacing=1, fontsize='large')
    print(f'Saving image to: output/visualizations/{Path(segmented_images[index]).stem}')
    plt.savefig(f'output/visualizations/{Path(segmented_images[index]).stem}.png',
                bbox_inches='tight')


def colorize(image_path, cmap, labels):
    """Apply a colormap to the grayscale prediction"""
    for path in image_path:
        prediction = Image.open(path)
        prediction = np.array(prediction).astype(np.uint8)
        prediction = prediction.squeeze().astype(np.uint8)
        pred_pil = Image.fromarray(prediction)
        pred_pil.putpalette(np.array(cmap).flatten().tolist())

        #image = Image.open(image_path)
        #images = [image, pred_pil]
        #widths, heights = zip(*(i.size for i in images))

        #total_width = sum(widths)
        #max_height = max(heights)

        #new_im = Image.new('RGB', (total_width, max_height))

        # Horizontally concatenate raw and predicted image
        #x_offset = 0
        #for im in images:
        #    new_im.paste(im, (x_offset,0))
        #    x_offset += im.size[0]

        # Label the predicted segmentation map
        colors = [ [color[0] / 255, color[1] / 255, color[2] / 255] 
                for color in cmap]
        labels = [label for label in labels]
        plt.figure(figsize = (9,9))
        ax = plt.imshow(pred_pil)
        plt.title(Path(path).stem)
        plt.axis('off')
        patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=labels[i]) ) for i in range(len(labels)) ]
        plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0, ncol=2, 
                handleheight=1.1, handlelength=2.3, labelspacing=1, columnspacing=0.8, fontsize='large')

        Path('output/visualizations').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'output/visualizations/{Path(path).stem}.png', bbox_inches='tight')    
        plt.close() # Save memory


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default='output/example_output/inference',
                        help='relative path to images being visualized, ex: images/test.png')
    parser.add_argument('--label_image', action='store_true', default=False,
                        help='')
    parser.add_argument('--colorize', action='store_true', default=False,
                        help='')
    
    args = parser.parse_args()

    path_to_images = os.path.join(args.images_path, '*.*')
    print(path_to_images)
    segmented_images_path = glob.glob(path_to_images)

    # Rellis-3D Color Palette
    rellis3d = {
        0: {"color": [0, 0, 0],  "name": "void"},
        1: {"color": [108, 64, 20],   "name": "dirt"},
        3: {"color": [0, 102, 0],   "name": "grass"},
        4: {"color": [0, 255, 0],  "name": "tree"},
        5: {"color": [0, 153, 153],  "name": "pole"},
        6: {"color": [0, 128, 255],  "name": "water"},
        7: {"color": [0, 0, 255],  "name": "sky"},
        8: {"color": [255, 255, 0],  "name": "vehicle"},
        9: {"color": [255, 0, 127],  "name": "object"},
        10: {"color": [64, 64, 64],  "name": "asphalt"},
        12: {"color": [255, 0, 0],  "name": "building"},
        15: {"color": [102, 0, 0],  "name": "log"},
        17: {"color": [204, 153, 255],  "name": "person"},
        18: {"color": [102, 0, 204],  "name": "fence"},
        19: {"color": [255, 153, 204],  "name": "bush"},
        23: {"color": [170, 170, 170],  "name": "concrete"},
        27: {"color": [41, 121, 255],  "name": "barrier"},
        31: {"color": [134, 255, 239],  "name": "puddle"},
        33: {"color": [99, 66, 34],  "name": "mud"},
        34: {"color": [110, 22, 138],  "name": "rubble"}}
    
    # LPCVC Color Pallette
    lpcvc_colors = [
        [213, 214, 212], # background
        [0, 102, 0], 
        [0, 255, 0], 
        [0, 153, 153], 
        [0, 128, 255], 
        [0, 0, 255], 
        [255, 255, 0], 
        [255, 0, 127], 
        [64, 64, 64], 
        [255, 0, 0], 
        [102, 0, 0], 
        [204, 153, 255],
        [102, 0, 204], 
        [255, 153, 204]
    ]
    lpcvc_labels = [
        'background', 
        'avalanche', 
        'building_undamaged', 
        'building_damaged', 
        'cracks/fissure/subsidence', 
        'debris/mud/rock_flow', 
        'fire/flare', 
        'flood/water/river/sea',
        'ice_jam_flow', 
        'lava_flow', 
        'person', 
        'pyroclastic_flow', 
        'road/railway/bridge', 
        'vehicle'
    ]

    if args.label_image:
        label_segmented_image(segmented_images_path, rellis3d)
    elif args.colorize:
        colorize(segmented_images_path, lpcvc_colors, lpcvc_labels)
