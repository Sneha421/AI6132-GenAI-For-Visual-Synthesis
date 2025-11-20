import json
import os
from groq import Groq

# Set the environment variable with GROQ API KEY
os.environ["GROQ_API_KEY"] = "fill_with_API_key"

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def transform_caption(original_caption):
    """Transform long LLAVA caption into concise InstructPix2Pix/ControlNet format"""

    prompt = f"""Convert this makeup description into a concise instruction for image editing (20-30 words).

Requirements:
- Focus only on makeup application steps
- Structure: [foundation/base] + [brows] + [lips] + [overall style]
- Use imperative verbs (Apply, Add, Create, Use, Achieve, etc.)
- Keep it natural and varied
- NO mention of "before/after" or image descriptions
- Just the instruction, nothing else

Original description: {original_caption}

Concise instruction:"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",  # Fast and capable model
        temperature=0.8,  # Adds variety to outputs
        max_tokens=100,
    )

    return chat_completion.choices[0].message.content.strip()

def process_caption_file(input_file, output_file):
    """Process entire JSON file of captions

    Input Data Format: [
            {"image_0": "path/to/img0.jpg", "image_1": "path/to/img1.jpg", conversations: {contains caption} },
            ...
        ]
    
    """

    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Processing {len(data)} captions...")

    # Transform each caption
    for i, item in enumerate(data):
        if 'conversations' in item:
            original_value = item['conversations'][1]['value']

            try:
                # Transform the caption
                new_caption = transform_caption(original_value)
                item['conversations'][1]['value'] = new_caption

            except Exception as e:
                print(f"Error processing {item['id']}: {e}")
                continue

    # Save output
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nCompleted! Saved to {output_file}")


if __name__ == "__main__":

    input_file = "makeup_transformation_instructions.json"
    output_file = "captions_transformed.json"

    process_caption_file(input_file, output_file)

