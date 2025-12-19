from PIL import Image
import torch
from torchvision import transforms
import os
from glob import glob

model_name = ['BiRefNet', 'BiRefNet_HR', 'BiRefNet_HR-matting'][0]

# # Option 1: loading BiRefNet with weights:
from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/{}'.format(model_name), trust_remote_code=True)

def mask_images():
    # Load Model
    device = 'cuda'
    torch.set_float32_matmul_precision(['high', 'highest'][0])

    birefnet.to(device)
    birefnet.eval()
    print('BiRefNet is ready to use.')
    birefnet.half()

    # Input Data
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024) if '_HR' not in model_name else (2048, 2048)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    import os
    from glob import glob

    src_dir = 'input_images'
    image_paths = glob(os.path.join(src_dir, '*'))
    dst_dir = 'masks'
    os.makedirs(dst_dir, exist_ok=True)
    for image_path in image_paths:
        print('Processing {} ...'.format(image_path))
        image = Image.open(image_path)
        input_images = transform_image(image).unsqueeze(0).to('cuda')
        input_images = input_images.half()

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        # Save Results
        file_ext = os.path.splitext(image_path)[-1]
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil = pred_pil.resize(image.size)
        pred_pil.save(image_path.replace(src_dir, dst_dir).replace(file_ext, '-mask.png'))