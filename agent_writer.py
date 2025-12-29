import os
import re
from google import genai
from google.genai import types
from PIL import Image

# ⚠️ WARNING: BEST PRACTICE is to use environment variables (os.getenv)
# NEVER commit a file containing your actual API key to a public repository.
# --- Configuration ---
# 1. REPLACE THIS PLACEHOLDER with your actual Gemini API Key
GEMINI_API_KEY = "" # Your API Key

# 2. Set your image path and desired output file name
IMAGE_PATH = "images/maze_inverted.png"
OUTPUT_FILE = "agent_solution.py"

# --- Code Context Modules (Easily Modifiable) ---
# Add paths to other Python files the model needs to reference.
CONTEXT_MODULE_PATHS = ["agent_solution.py"] 
# The actual file paths in your environment go here.

# --- System Instruction (General Context) ---
SYSTEM_INSTRUCTION = (
    "You are an expert Python programmer specializing in image processing and computer vision. "
    "Your response should be the final, complete, self-contained Python script to solve the maze. "
    "If context modules are provided, assume their logic is available. "
    "Output MUST be a single Python markdown code block (```python ... ```) with no extra text."
)

# --- Helper Function for Code Extraction ---

def extract_python_code(text: str) -> str:
    """
    Extracts the content of the first Python Markdown code block 
    (```python ... ```) from the model's response text.
    """
    # Regex to find ```python followed by content, up to the closing ```
    match = re.search(r"```python\s*\n(.*?)\s*```", text, re.DOTALL)
    
    if match:
        # Return the captured group (the content inside the code block)
        return match.group(1).strip()
    else:
        # If no code block is found, return the original text as a fallback
        return text.strip()

# --- Main Logic Function ---

def generate_code_from_image_and_modules(api_key: str, image_path: str, output_file: str, module_paths: list[str], system_instruction: str):
    """
    Initializes the Gemini client, loads an image, reads code modules, 
    and injects all context into the prompt for code generation.
    """

    try:
        # 1. Initialize the client
        client = genai.Client(api_key=api_key)
        
        # 2. Load the image using PIL
        img = Image.open(image_path)

        # 3. Read and format context code modules
        context_code_parts = []
        for path in module_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    content = f.read()
                
                # Format the context so the model can easily identify it
                context_code_parts.append(
                    f"\n--- Context Module: {os.path.basename(path)} ---\n"
                    f"{content}\n"
                    f"------------------------------------------\n"
                )
            else:
                print(f"⚠️ Warning: Context module file not found at '{path}'. Skipping.")

        full_context = "\n".join(context_code_parts)
        
        # 4. Define the complete prompt (Context + Image Task)
        prompt = (
            f"{full_context}\n"
            "***\n"
            "Your task is to solve the following maze image and generate the final python code.\n"
            "This is an image of a maze. " 
            "The start point is the red dot and the end point is the green dot." 
            "Overlay a visible blue line from the start to the end." 
            "Make the solution path traverse in the middle of the path rather than hugging the walls." 
            f"The input image path is '{IMAGE_PATH}'."
            "The output path should be 'images/maze_solution.png'." 
            "Write the final python code to solve this, potentially referencing the provided context modules."
        )

        # 5. Define the generation configuration with the system instruction
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
        
        # 6. Call the API (passing prompt and image)
        print("💡 Generating code from image and context modules, please wait...")
        # The prompt contains all text context (user instructions + code modules)
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[prompt, img], 
            config=config
        )
        
        # 7. Extract and clean the code
        generated_text = response.text
        code_only = extract_python_code(generated_text)
        
        if not code_only or "import" not in code_only:
            print("❌ Error: Could not reliably extract complete Python code. Check the full response below:")
            print("-" * 30)
            print(generated_text)
            print("-" * 30)
            return

        # 8. Save the extracted code to the output file
        with open(output_file, "w") as f:
            f.write(code_only)
        
        print(f"✅ Success! Generated Python code saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: The primary image file not found at '{image_path}'. Please check the path.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# --- Execute the main function ---
if __name__ == "__main__":
    generate_code_from_image_and_modules(
        api_key=GEMINI_API_KEY, 
        image_path=IMAGE_PATH, 
        output_file=OUTPUT_FILE, 
        module_paths=CONTEXT_MODULE_PATHS, # Pass the list of file paths
        system_instruction=SYSTEM_INSTRUCTION
    )