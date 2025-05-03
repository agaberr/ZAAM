import requests
from bs4 import BeautifulSoup
import re
import json
import time
from googleapiclient.discovery import build
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = "AIzaSyBA-2V_K-sjI9voApCARrlKXZDdVgg5OPM"
SEARCH_ENGINE_ID = "c7df448be45934f6e"

ALLOWED_DOMAINS = ["allrecipes.com"]

class RecipeExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_from_url(self, url):
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                html_content = response.text
                
                ingredients = self.extract_recipe_ingredients(html_content)
                instructions = self.extract_instructions(html_content)
                title = self.extract_title(html_content)
            
                return {
                    "url": url,
                    "title": title,
                    "ingredients": ingredients,
                    "instructions": instructions
                }
            else:
                return {"error": f"Failed to retrieve content: HTTP {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Error extracting recipe: {str(e)}"}
    
    def extract_recipe_ingredients(self, html_content):

        soup = BeautifulSoup(html_content, 'html.parser')
        
        ingredients = []
        
        ingredients_list = soup.find('ul', {'class': 'mm-recipes-structured-ingredients__list'})
        
        if ingredients_list:
            list_items = ingredients_list.find_all('li', {'class': 'mm-recipes-structured-ingredients__list-item'})
            
            for item in list_items:
                quantity_span = item.find('span', {'data-ingredient-quantity': 'true'})
                unit_span = item.find('span', {'data-ingredient-unit': 'true'})
                name_span = item.find('span', {'data-ingredient-name': 'true'})
                
                ingredient_parts = []
                
                if quantity_span and quantity_span.text.strip():
                    ingredient_parts.append(quantity_span.text.strip())
                
                if unit_span and unit_span.text.strip():
                    ingredient_parts.append(unit_span.text.strip())
                
                if name_span and name_span.text.strip():
                    ingredient_parts.append(name_span.text.strip())
                
                if ingredient_parts:
                    ingredient_text = " ".join(ingredient_parts).strip()
                    ingredients.append(ingredient_text)
                else:
                    p_tag = item.find('p')
                    if p_tag:
                        ingredients.append(p_tag.text.strip())
        
        if not ingredients:
            ingredient_spans = soup.find_all('span', {'data-ingredient-name': 'true'})
            for span in ingredient_spans:
                parent_p = span.find_parent('p')
                if parent_p:
                    ingredients.append(parent_p.text.strip())
                else:
                    ingredients.append(span.text.strip())
        
        return ingredients
    
    def extract_title(self, html_content):
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        title_element = soup.find('h1', {'class': 'recipe-title'})
        
        if not title_element:
            title_element = soup.find('h1')
        
        if title_element:
            return title_element.text.strip()
        else:
            return "Unknown Recipe"
    
    def extract_instructions(self,html_content):
   
        soup = BeautifulSoup(html_content, 'html.parser')
        
        instructions = []
        
        instructions_heading = soup.find('h2', {'class': 'mm-recipes-structured-ingredients__heading'})
        
        if instructions_heading:

            instructions_items = []
            
            parent_container = instructions_heading.parent
            if parent_container:
                instructions_items = parent_container.find_all('li', {'class': 'mntl-sc-block-group--LI'})
            
            if not instructions_items:
                instructions_items = soup.find_all('li', {'class': 'mntl-sc-block-group--LI'})
            
            # Extract text from each ingredient item
            for item in instructions_items:
                paragraphs = item.find_all('p', {'class': 'mntl-sc-block-html'})
                
                for p in paragraphs:
                    # Clean up the text and add to ingredients list
                    instructions_text = p.text.strip()
                    if instructions_text and not instructions_text.startswith('Gather all ingredients'):
                        instructions.append(instructions_text)
        
        # If the above method didn't work, try finding ingredients by specific classes or patterns
        if not instructions:
            instructions_items = soup.find_all('li', {'class': 'ingredients-item'})
            for item in instructions_items:
                instructions_text = item.text.strip()
                if instructions_text:
                    instructions.append(instructions_text)
        
        return instructions
    
def google_search(query, num_results=10):
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        
        site_restriction = "site:allrecipes.com"
        site_restricted_query = f"{query} recipe {site_restriction}"
        
        print(f"Searching for: {site_restricted_query}")
        
        result = service.cse().list(
            q=site_restricted_query, 
            cx=SEARCH_ENGINE_ID, 
            num=num_results
        ).execute()
        
        if "items" not in result:
            print(f"No search results returned. Response: {result}")
            return []
        
        search_results = []
        for item in result["items"]:
            url = item["link"]
            title = item["title"]
            print(f"Found: {title} at {url}")
            search_results.append({"url": url, "title": title})
        
        return search_results
    
    except Exception as e:
        print(f"Error performing Google search: {str(e)}")
 
        return []
  
def get_recipes_for_query(query, max_results=2):
    search_results = google_search(query, max_results * 2)
    # print(f"Total search results: {len(search_results)}")
    # print(f"Search results: {search_results}")

    if not search_results:
        return []
    
    extractor = RecipeExtractor()
    recipes = []
    
    for result in search_results[:max_results]:
        time.sleep(1)
        
        recipe = extractor.extract_from_url(result["url"])
        
        if recipe and "error" not in recipe:
            if recipe["ingredients"] and recipe["instructions"]:
                recipes.append(recipe)
    
    return recipes

def transform_recipe_to_conversation(recipe):

    title = recipe.get('title', 'this dish')
    ingredients = recipe.get('ingredients', [])
    instructions = recipe.get('instructions', [])
    
    conversation = f"Hey! So I wanted to tell you about this amazing {title}. It's super tasty and pretty easy to make.\n\n"
    
    conversation += "For this recipe, you'll need to grab:\n"
    for ingredient in ingredients:
        conversation += f"- {ingredient}\n"
    
    conversation += f"\nOkay, so here's how to make the {title}:\n\n"
    
    combined_instructions = ""
    for i, instruction in enumerate(instructions):
        if i == 0:
            combined_instructions += f"First, {instruction.lower()} "
        elif i == len(instructions) - 1:
            combined_instructions += f"Finally, {instruction.lower()} "
        else:
            transitions = ["Then", "After that", "Next", "Once that's done"]
            transition = transitions[i % len(transitions)]
            combined_instructions += f"{transition}, {instruction.lower()} "
    
    conversation += combined_instructions.strip() + "\n\n"
    
    conversation += f"The {title} is ready! I like to pair it with a simple side salad or some kimchi if you have it. The flavor is so good - a perfect balance of sweet and savory with just a hint of spice. Let me know how it turns out if you try it!"
    
    return conversation

def transform_all_recipes(recipes_list):

    conversational_guides = []
    
    for recipe in recipes_list:
        conversational_guide = transform_recipe_to_conversation(recipe)
        conversational_guides.append({
            'title': recipe.get('title', 'Recipe'),
            'conversation': conversational_guide
        })
    
    return conversational_guides




