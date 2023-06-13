import cv2
from datetime import datetime
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch

start = datetime.now()

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

image = cv2.imread("truck.jpg")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

end = datetime.now()
print((end - start).total_seconds())
