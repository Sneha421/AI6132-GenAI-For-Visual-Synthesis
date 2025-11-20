
"""
Generate synthetic instruction-following dataset for makeup transformation
This creates training data from image pairs (image 0 -> image 1)
Uses batched inference for efficiency
"""

# Load necessary libraries
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
import random
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import os

# Set environment variables for performance and stability
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MakeupTransformationDatasetGenerator:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", batch_size=8):
        """Initialize LLaVA model for generating transformation descriptions"""
        print(f"Loading {model_name}...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = batch_size

    def generate_transformation_prompts(self):
        """Create diverse instruction prompts for makeup transformation (image 0 -> image 1)"""
        prompts = [
            "Look at these two images. Describe the makeup transformation from the first image to the second image. What changed?",

            "Compare the makeup in both images. What steps would you take to transform the makeup look from image 1 to image 2?",

            "Analyze the differences in makeup between these two images. Describe what makeup techniques and products you would need to recreate this transformation.",

            "What are the key makeup changes from the first to the second image? List the specific steps to achieve this transformation.",

            "Describe how to go from the makeup look in the first image to the makeup look in the second image. Be specific about colors, techniques, and application.",

            "Create a step-by-step makeup tutorial to transform from the look in image 1 to the look in image 2.",

            "What makeup products and techniques would you change to go from the first look to the second look? Describe the transformation process.",

            "Compare these two makeup looks. What's different about the eyes, lips, face makeup, and overall style? How would you recreate this change?",

            "Describe the before and after makeup transformation. What specific changes were made and how would you replicate them?",

            "If you wanted to transform your makeup from the first image to the second image, what would you do differently? Provide detailed instructions.",
        ]
        return prompts

    def prepare_batch_inputs(self, image_pairs: List[Tuple[str, str]], prompts: List[str]):
        """
        Prepare batched inputs for LLaVA model
        Each input consists of two images - bare-faced and makeup-applied
        """
        images_list = []
        texts_list = []

        for (img0_path, img1_path), prompt in zip(image_pairs, prompts):
            # Load both images
            img0 = Image.open(img0_path).convert("RGB").resize((256, 256), Image.LANCZOS) # bare-faced
            img1 = Image.open(img1_path).convert("RGB").resize((256, 256), Image.LANCZOS) # makeup-applied


            # LLaVA can handle multiple images in sequence
            images_list.append([img0, img1])

            # Create conversation with two images
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # First image
                        {"type": "image"},  # Second image
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            # Apply chat template per LLAVA requirements
            prompt_text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            texts_list.append(prompt_text)

        return images_list, texts_list

    def generate_batch_descriptions(self, image_pairs: List[Tuple[str, str]], prompts: List[str]):
        """Generate descriptions for a batch of image pairs"""
        images_list, texts_list = self.prepare_batch_inputs(image_pairs, prompts)

        # Flatten images for processor
        flat_images = []
        for img_pair in images_list:
            flat_images.extend(img_pair)

        # Process batch
        inputs = self.processor(
            images=flat_images,
            text=texts_list,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device, torch.float16)

        # Generate responses for batch
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300, # Descriptive output
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,  # Greedy for batch efficiency
        )

        # Decode responses
        responses = []
        for output in outputs:
            generated_text = self.processor.decode(output, skip_special_tokens=True)
            # Extract just the response after "ASSISTANT:"
            response = generated_text.split("ASSISTANT:")[-1].strip()
            responses.append(response)

        return responses

    def load_image_pairs(self, pairs_file):
        """
        Load image pairs from a file
        Expected format (JSON):
        [
            {"image_0": "path/to/img0.jpg", "image_1": "path/to/img1.jpg"},
            ...
        ]
        """

        with open(pairs_file, 'r') as f:
            data = json.load(f)
            return [(d['image_0'], d['image_1']) for d in data]


    def create_dataset(self, image_pairs_file, image_dir, output_file, num_prompts_per_pair=1):
        """
        Generate full transformation dataset from your image pairs

        Args:
            image_pairs_file: JSON/CSV file with image pairs
            image_dir: Base directory containing images
            output_file: JSON file to save the dataset
            num_prompts_per_pair: How many different prompts per image pair
        """
        image_dir = Path(image_dir)
        image_pairs = self.load_image_pairs(image_pairs_file)

        print(f"Processing {len(image_pairs)} image pairs...")

        dataset = []
        all_prompts = self.generate_transformation_prompts()

        # Process in batches
        for i in tqdm(range(0, len(image_pairs), self.batch_size)):
            batch_pairs = image_pairs[i:i + self.batch_size]

            # Generate multiple prompts per pair
            for prompt_idx in range(num_prompts_per_pair):
                # Select random prompts for this batch
                batch_prompts = [random.choice(all_prompts) for _ in batch_pairs]

                # Prepare full paths
                batch_pairs_full = [
                    (str(image_dir / p0), str(image_dir / p1))
                    for p0, p1 in batch_pairs
                ]

                try:
                    # Generate batch responses
                    responses = self.generate_batch_descriptions(batch_pairs_full, batch_prompts)

                    # Create dataset entries
                    for (img0, img1), prompt, response in zip(batch_pairs, batch_prompts, responses):
                        data_entry = {
                            "id": f"makeup_transform_{len(dataset)}",
                            "image_0": str(img0),
                            "image_1": str(img1),
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": f"<image>\n<image>\n{prompt}"  # Two images
                                },
                                {
                                    "from": "LLAVA",
                                    "value": response
                                }
                            ]
                        }
                        dataset.append(data_entry)

                except Exception as e:
                    print(f"\nError processing batch at index {i}: {e}")
                    continue

        # Save dataset
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"\nDataset created! {len(dataset)} transformation samples saved to {output_file}")
        return dataset


if __name__ == "__main__":
    # Initialize generator with batch size
    generator = MakeupTransformationDatasetGenerator(
        model_name="llava-hf/llava-1.5-7b-hf",
        batch_size=4  # Process 4 image pairs at once. Higher batch sizes result in more GPU memory usage.
    )


    # Generate dataset
    dataset = generator.create_dataset(
        image_pairs_file="image_pairs.json",  # Image pairs file of the format [ {"image_0": "path/to/img0.jpg", "image_1": "path/to/img1.jpg"} ]
        image_dir="ffhq-makeup/FFHQ-Makeup",  # FFHQ image directory with 90k images
        output_file="makeup_transformation_instructions.json", # Output file with the same format as input & a prompt attribute
        num_prompts_per_pair=1  # Generate 1 prompt per image pair
    )

