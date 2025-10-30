# AI6132-GenAI-For-Visual-Synthesis
This repository is to track the code for the final project for this course

### Reference Materials
1. FFHQ-Makeup: Paired Synthetic Makeup Dataset with Facial Consistency Across Multiple Styles (dataset avail at
https://huggingface.co/datasets/cyberagent/FFHQ-Makeup)
2. BeautyBank: Encoding Facial Makeup in Latent Space (dataset avail at
https://huggingface.co/datasets/lulululululululululu/Bare-Makeup-Synthesis-Dataset)
3. Face-MakeUp: Multimodal Facial Prompts for Text-to-Image Generation
https://arxiv.org/pdf/2501.02523
4. InstructPix2Pix
https://arxiv.org/abs/2312.04780

### Tentative Plan
Current plan:
1. Use part of FFHQ-makeup dataset and bare beauty dataset to generate instructions using a vision language transformer
    1. Can use LLAVA for this
    2. Few shot prompting
    3. Thus, we are using 2 datasets and are also creating our own
2. Use InstructPix2Pix and ControlNet on the generated dataset to go from bare face to make up
3. Compare CLIP - This is mainly used for text to image similarity. Is your generated image matching the t instructions that you gave
    1.  FID, CLIP & LPIPS for GT image and generated makeup image similarity

