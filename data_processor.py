import json
import os
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class GPTContentProcessor:
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key"""
        openai.api_key = api_key
        self.client = OpenAI()
        
    def process_content(self, markdown_content: str) -> Dict[str, Any]:
        """Process markdown content using GPT-4"""
        try:
            # System prompt to define the structure
            system_prompt = """
            You are a content structuring assistant. Convert the provided markdown web content into a structured JSON format.
            Follow these rules:
            1. Extract main sections and subsections
            2. Preserve hierarchical relationships
            3. Extract all links with their text and URLs
            4. Identify navigation elements
            5. Separate content into logical chunks
            
            Use this JSON structure:
            {
                "title": "page title",
                "content": "main content text if available",
                "links": [
                    {
                        "text": "link text and description if available",
                        "url": "link url",
                    }
                ]
            }
            """

            # User prompt with the content
            user_prompt = f"""
            Please convert this markdown content into structured JSON:
            {markdown_content}
            """

            # Make API call to GPT-4
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=16000,
            )
            print(response)
            # Parse the response
            json_content = response.choices[0].message.content
            if json_content.startswith('```json'):
                json_content = json_content.replace('```json\n', '', 1)
                json_content = json_content.replace('\n```', '', 1)
            structured_content = json.loads(json_content)
            return structured_content

        except Exception as e:
            print(f"Error processing content with GPT: {e}")
            print(f"Response content: {response.choices[0].message.content}")
    
    def process_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """Process a markdown file and return structured content"""
        try:
            # Read markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)

            # Process with GPT-4
            structured_data = self.process_content(content)
            
            # Save processed data
            output_path = f"{os.path.splitext(file_path)[0]}_structured.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)

            return structured_data

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

    def batch_process_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple markdown files"""
        processed_files = []
        
        for file_path in file_paths:
            result = self.process_markdown_file(file_path)
            if result:
                processed_files.append(result)
                print(f"Successfully processed: {file_path}")
            else:
                print(f"Failed to process: {file_path}")
                
        return processed_files


def main():
    # Initialize processor with your API key
    api_key = load_dotenv()
    # api_key = os.getenv("OPENAI_API_KEY")
    processor = GPTContentProcessor(api_key=api_key)
    
    # Example usage
    markdown_files = ['data.md']
    
    # Process all files
    processed_data = processor.batch_process_files(markdown_files)
    print(processed_data)
    
    # Save combined results
    with open('all_processed_data.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()