import re

# Function to extract content from square brackets
def extract_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        input_strings = file.readlines()

    pattern = r"\[([^\]]+)\]"
    extracted_parts = [re.search(pattern, line).group(1) for line in input_strings if re.search(pattern, line)]
    
    return extracted_parts

# Function to save extracted parts to a new file
def save_to_file(output_path, extracted_parts):
    with open(output_path, "w", encoding='utf-8') as file:
        for item in extracted_parts:
            file.write(item + '\n')

# Example usage
input_path = 'asha_bhosale.txt'  # Input file path
output_path = 'asha_bhosale.txt'  # Output file path

extracted_parts = extract_from_file(input_path)
save_to_file(output_path, extracted_parts)

print(f"Extracted parts saved to {output_path}")
