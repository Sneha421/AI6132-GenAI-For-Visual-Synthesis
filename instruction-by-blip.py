from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import torch, json

# ======================================================
# Load model
# ======================================================
MODEL_ID = "Salesforce/instructblip-flan-t5-xl"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map={"": device}
)

# ======================================================
# Prompt
# ======================================================
BASE_PROMPT = (
    "You are an image editing assistant that writes one-sentence commands for applying makeup. "
    "Look at the given image and imagine transforming a bare face into this makeup look. "
    "Write exactly one concise English command that starts with 'Add' or 'Apply'. "
    "Include color adjectives for visible features such as lipstick, eyeshadow, eyeliner, and blush. "
    "Mention at least three features if visible. "
    "Avoid style or mood phrases like 'for a natural look'. "
    "Output only the command sentence itself.\n\n"
    "Examples:\n"
    "Add pink lipstick, light brown eyeshadow, and soft blush.\n"
    "Apply red lipstick, gold eyeshadow, and black eyeliner.\n"
    "Add coral lipstick, brown eyeshadow, black eyeliner, and pink blush.\n\n"
    "Now write the command for this image:"
)


# ======================================================
# Instruction Generation Function
# ======================================================
def generate_instruction(image_path: Path):
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        return None

    inputs = processor(images=image, text=BASE_PROMPT, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=70,
        do_sample=True,
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.05
    )
    text = processor.decode(out[0], skip_special_tokens=True).strip()
    return text


# ======================================================
# Batch Generation
# ======================================================
ROOT = Path("/kaggle/input/ffhq-sorted/compiled_output_all")
bare_dir = ROOT / "bare"
makeup_styles = [f"makeup_0{i}" for i in range(4, 6)]  # makeup_01 ~ makeup_05
OUTPUT_DIR = Path("/kaggle/working/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bare_imgs = sorted(list(bare_dir.glob("*.jpg")))
print(f" Found {len(bare_imgs)} bare images.")

for style in makeup_styles:
    results = []
    makeup_dir = ROOT / style
    output_file = OUTPUT_DIR / f"instruction_{style}.jsonl"

    for bare_img in bare_imgs:
        makeup_path = makeup_dir / f"{bare_img.stem.replace('bare', style)}.jpg"
        if not makeup_path.exists():
            continue
        text = generate_instruction(makeup_path)
        if text:
            results.append({
                "instruction": text,
                "input": str(bare_img.relative_to(ROOT)),
                "target": str(makeup_path.relative_to(ROOT))
            })
            print(f"{bare_img.name} [{style}] → {text}")
        else:
            print(f"Skipped {bare_img.name} [{style}]")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    print(f"\n {style}: {len(results)} results saved to {output_file}\n")

print("All done!")
