import os
import glob
import json
import re
from openai import OpenAI
import google.generativeai as genai

# Initialize OpenAI Client
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = None
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)

# Initialize Google AI Client
google_api_key = os.environ.get("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

# Determine which LLM provider to use (default to OpenAI if both are set)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()  # "openai" or "gemini"

def generate_mermaid_from_text():
    """Workflow A: Watches source/drafts/*.txt -> Creates source/mermaid/*.mmd"""
    drafts = glob.glob("source/drafts/*.txt")

    try:
        system_prompt = open("prompts/mermaid-generator.md", "r").read()
    except FileNotFoundError:
        print("Error: prompts/mermaid-generator.md not found.")
        return

    for draft_path in drafts:
        base_name = os.path.basename(draft_path).replace(".txt", "")
        output_path = f"source/mermaid/{base_name}.mmd"

        # Idempotency check
        if os.path.exists(output_path):
            print(f"Skipping {base_name} (Exists)")
            continue

        print(f"Generating Mermaid for: {base_name} using {LLM_PROVIDER}...")
        user_content = open(draft_path, "r").read()

        try:
            content = None

            if LLM_PROVIDER == "gemini":
                if not google_api_key:
                    print(f"Error: GOOGLE_API_KEY not set. Cannot use Gemini.")
                    continue

                # Use Gemini for text generation
                model = genai.GenerativeModel("gemini-2.0-flash-exp")
                full_prompt = f"{system_prompt}\n\n{user_content}"
                response = model.generate_content(full_prompt)
                content = response.text
            else:
                # Use OpenAI (default)
                if not openai_client:
                    print(f"Error: OPENAI_API_KEY not set. Cannot use OpenAI.")
                    continue

                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                )
                content = response.choices[0].message.content

            # Extract Mermaid code from response
            mermaid_code = re.search(r"```mermaid\n(.*?)\n```", content, re.DOTALL)

            if mermaid_code:
                with open(output_path, "w") as f:
                    f.write(mermaid_code.group(1))
                print(f"Saved {output_path}")
            else:
                print(f"Failed to parse Mermaid code for {base_name}")
        except Exception as e:
            print(f"API Error for {base_name}: {e}")

def generate_images():
    """Workflow B & C: Watches source/generative & polished -> Creates assets/diagrams-generated/*.png"""
    json_files = glob.glob("source/generative/*.json") + glob.glob("source/polished/*.json")

    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {json_path}")
            continue

        if "output_image" in meta:
            output_path = meta["output_image"]
        else:
            base_name = os.path.basename(json_path).replace(".json", ".png")
            output_path = f"assets/diagrams-generated/{base_name}"

        if os.path.exists(output_path):
            print(f"Skipping Image {output_path} (Exists)")
            continue

        # Determine which model to use (default to dall-e-3)
        image_model = meta.get("image_model", "dall-e-3")

        print(f"Generating Image for: {json_path} using {image_model}...")
        final_prompt = meta.get("prompt", "")

        # Handle Workflow C (Polished)
        if "source_logic" in meta:
            try:
                with open(meta["source_logic"], "r") as mmd:
                    mermaid_content = mmd.read()
                style_prompt = meta.get("style_prompt", "")
                final_prompt = f"Create a diagram based on this logic:\n{mermaid_content}\n\nStyle Guide:\n{style_prompt}"
            except FileNotFoundError:
                print(f"Error: Source logic file {meta['source_logic']} not found.")
                continue

        try:
            if image_model == "imagen-3.0-generate-001":
                # Use Google Imagen 3
                if not google_api_key:
                    print(f"Error: GOOGLE_API_KEY not set. Cannot use {image_model}")
                    continue

                model = genai.GenerativeModel(image_model)
                response = model.generate_content(final_prompt)

                # Save the generated image
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data'):
                            img_data = part.inline_data.data
                            with open(output_path, "wb") as handler:
                                handler.write(img_data)
                            print(f"Saved Image: {output_path}")
                            break
                else:
                    print(f"No image data returned from Imagen 3 for {json_path}")
            else:
                # Use DALL-E-3 (default)
                response = openai_client.images.generate(
                    model="dall-e-3",
                    prompt=final_prompt,
                    size="1024x1024",
                    quality="hd",
                    n=1,
                )
                image_url = response.data[0].url

                import requests
                img_data = requests.get(image_url).content
                with open(output_path, "wb") as handler:
                    handler.write(img_data)
                print(f"Saved Image: {output_path}")

        except Exception as e:
            print(f"Failed to generate image for {json_path}: {e}")

if __name__ == "__main__":
    # Check if at least one API key is configured
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_google = bool(os.environ.get("GOOGLE_API_KEY"))

    if not has_openai and not has_google:
        print("Error: No API keys found. Please set OPENAI_API_KEY and/or GOOGLE_API_KEY.")
        exit(1)

    print(f"Active LLM Provider: {LLM_PROVIDER}")

    if LLM_PROVIDER == "openai" and not has_openai:
        print("Warning: LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is not set.")

    if LLM_PROVIDER == "gemini" and not has_google:
        print("Warning: LLM_PROVIDER is set to 'gemini' but GOOGLE_API_KEY is not set.")

    if not has_openai:
        print("Note: OPENAI_API_KEY not set. OpenAI features will not be available.")

    if not has_google:
        print("Note: GOOGLE_API_KEY not set. Google AI features (Gemini, Imagen 3) will not be available.")

    generate_mermaid_from_text()
    generate_images()
