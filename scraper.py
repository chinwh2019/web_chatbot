# write a class that will scrape the data from the website and save it to a file in the specified format.

import json
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin

import markdown
import requests
from bs4 import BeautifulSoup
from html2text import HTML2Text


class StructuredContent:
    def __init__(self, page_title: str, navigation_items: List):
        self.page_title = page_title
        self.navigation_items = navigation_items

class MenuItem:
    def __init__(self, title: str, url: str, sub_items: List):
        self.title = title
        self.url = url
        self.sub_items = sub_items


class Scrapper:
    def __init__(self, input_file: str):
        self.input_file = input_file

    def fetch_and_save_webpage(self, url: str, output_file: str) -> None:
        """
        Fetches webpage content and saves it as markdown
        
        Args:
            url: The webpage URL to fetch
            output_file: Path to save the markdown file
        """
        try:
            # Fetch webpage content
            response = requests.get(url)
            response.raise_for_status()
            
            # Convert HTML to markdown
            h2t = HTML2Text()
            h2t.ignore_links = False
            h2t.ignore_images = False
            h2t.ignore_tables = False
            h2t.body_width = 0  # Disable line wrapping
            
            markdown_content = h2t.handle(response.text)
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
                
            self.input_file = output_file  # Update input file path
            
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch webpage: {str(e)}")
        except IOError as e:
            raise Exception(f"Failed to save markdown file: {str(e)}")

    def clean_text(self, text):
        return text.strip()

    def process_menu_item(self, li):
        title = ""
        url = ""
        sub_items = []
        a_tag = li.find('a')
        if a_tag:
            title = self.clean_text(a_tag.text)
            url = a_tag.get('href')
            sub_ul = li.find('ul', recursive=False)
            if sub_ul:
                for sub_li in sub_ul.find_all('li', recursive=False):
                    sub_item = self.process_menu_item(sub_li)
                    if sub_item:
                        sub_items.append(sub_item)
        return MenuItem(title, url, sub_items)

    def scrape_data(self):
        with open(self.input_file, 'r') as file:
            markdown_content = file.read()
            html_content = markdown.markdown(markdown_content)
            soup = BeautifulSoup(html_content, 'html.parser')

            page_title = ""
            h1_tag = soup.find('h1')
            if h1_tag:
                page_title = self.clean_text(h1_tag.text)
            else:
                first_line = markdown_content.split('\n')[0]
                page_title = self.clean_text(first_line)

            navigation_items = []
            for ul in soup.find_all('ul', recursive=False):
                for li in ul.find_all('li', recursive=False):
                    menu_item = self.process_menu_item(li)
                    if menu_item:
                        navigation_items.append(menu_item)

            return StructuredContent(page_title, navigation_items)

    def convert_to_dict(self, structured_content: StructuredContent) -> Dict:
        def menu_item_to_dict(item: MenuItem) -> Dict:
            return {
                'title': item.title,
                'url': item.url,
                'sub_items': [menu_item_to_dict(sub_item) for sub_item in item.sub_items] if item.sub_items else []
            }

        return {
            'page_title': structured_content.page_title,
            'navigation_items': [menu_item_to_dict(item) for item in structured_content.navigation_items]
        }

    def save_structured_data(self, structured_data: Dict, output_file: str, format: str = 'json'):
        output_path = Path(output_file)

        if format == 'jsonl':
            def flatten_navigation(nav_items, parent_title=""):
                flattened = []
                for item in nav_items:

                    current_item = {
                        'title': item['title'],
                        'url': item['url'],
                        'parent': parent_title,
                        'full_path': f"{parent_title} > {item['title']}" if parent_title else item['title']
                    }
                    flattened.append(current_item)

                    if item['sub_items']:
                        flattened.extend(flatten_navigation(item['sub_items'], current_item['full_path']))
                return flattened
            
            flat_navigation = flatten_navigation(structured_data['navigation_items'])
            with open(output_path, 'w') as file:
                for item in flat_navigation:
                    file.write(json.dumps(item) + '\n')
        else:       
            with open(output_path, 'w') as file:
                json.dump(structured_data, file, indent=4)  # Save as JSON


if __name__ == "__main__":
    scrapper = Scrapper('')
    scrapper.fetch_and_save_webpage('https://r.jina.ai/https://network.mobile.rakuten.co.jp/support/', 'data.md')
    # structured_content = scrapper.scrape_data()   
    # structured_data = scrapper.convert_to_dict(structured_content)
    # scrapper.save_structured_data(structured_data, 'output.json', format='json')
    # scrapper.save_structured_data(structured_data, 'output.jsonl', format='jsonl')
