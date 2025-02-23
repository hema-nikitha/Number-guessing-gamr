import os
import matplotlib.pyplot as plt
from torchvision.models import vgg19, VGG19_Weights
from src import run_style_transfer
from src import image_loader, imshow

def generate_output(content_img_path, style_img_path):
    content_img = image_loader(content_img_path)
    style_img = image_loader(style_img_path)
    
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    input_img = content_img.clone() 
    output = run_style_transfer(cnn, content_img, style_img, input_img)

    imshow(output, title='Output Image', save=True)
    plt.show()
    plt.close()

if __name__ == "__main__":
    content_imgs = os.listdir("./data/content")
    for i, content_img in enumerate(content_imgs):
        print(f"[{i}] - {content_img}")
    while True:
        content_n = int(input("Select the content image number: "))
        try:
            content_img_path = os.path.join("./data/content/", content_imgs[content_n])
            print(f"Content image selected: {content_img_path}")
            break
        except Exception as e:
            print("Invalid Selection. Try again!")
    
    style_imgs = os.listdir("./data/styles")
    print("\nStyle Images::")
    for i, style_img in enumerate(style_imgs):
        print(f"[{i}] - {style_img}")
    while True:
        style_n = int(input("Select the style image number: "))
        try:
            style_img_path = os.path.join("./data/styles/", style_imgs[style_n])
            print(f"Style image selected: {style_img_path}")
            break
        except Exception as e:
            print("Invalid Selection. Try again!")
    
    generate_output(content_img_path, style_img_path)