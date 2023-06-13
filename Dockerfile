FROM python:3

RUN pip3 install torch torchvision torchaudio
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git
RUN pip3 install opencv-python-headless

WORKDIR /app

RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
RUN wget https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg

COPY sam.py .

CMD ["python", "sam.py"]
