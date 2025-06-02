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
    
    def extractUrl(self, url):
        
                    #### bsend el requet
            response = requests.get(url,
        headers= {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            },
              timeout=10)

    ######## lw success b3ml extractions ll main items ll recipe        
            if response.status_code == 200:

                    htmlContent = response.text
                    
                ## bnady extraction ingredient w instructions w title el2akla
                    ingredients = self.extractIngredients(htmlContent)
                    
                    instructions = self.extract_instructions(htmlContent)
                    
                    title = self.GETtitle(htmlContent)
                
                    return {
                        "url": url,
                        "title": title,
                        "ingredients": ingredients,
                        "instructions": instructions
                    }
            else:
                return "error"
                

    
    def extractIngredients(self, html_content):

        soup = BeautifulSoup(html_content, 'html.parser')
        
        ingr = []

        ### b3ml selection list bta3et ingrdient with class names
        
        ingredientsList = soup.find('ul', {'class': 'mm-recipes-structured-ingredients__list'})
        
        if ingredientsList:

            ### hena lw tl3 fi ingredients list b3ml extract ll items
            list_items = ingredientsList.find_all('li', {'class': 'mm-recipes-structured-ingredients__list-item'})
            
            for item in list_items:
                
                ###### blf 3la kol item w a3ml extract ll spans ely mwgoda
                quantitySpan = item.find('span', {'data-ingredient-quantity': 'true'})
                
                Upan = item.find('span', {'data-ingredient-unit': 'true'})

                # bageef span with this class name


                NAMEspan = item.find('span', {'data-ingredient-name': 'true'})
                
                ingredientparts = []
                
                if quantitySpan and quantitySpan.text.strip():
                    ingredientparts.append(quantitySpan.text.strip())
                
                if Upan and Upan.text.strip():
                    ingredientparts.append(Upan.text.strip())
                
                if NAMEspan and NAMEspan.text.strip():
                    ingredientparts.append(NAMEspan.text.strip())
                
                if ingredientparts:
                    ingredient_text = " ".join(ingredientparts).strip()
                    ingr.append(ingredient_text)
                else:
                   
                    if item.find('p'):
                        ingr.append(item.find('p').text.strip())
        
        if not ingr:

            ## hen lw tl3 mafessh ingredients list b3ml extract ll spans ely mwgoda

            spans = soup.find_all('span', {'data-ingredient-name': 'true'})
            for span in spans:

                if  span.find_parent('p'):

                    ingr.append( span.find_parent('p').text.strip())
                    # print( span.find_parent('p'))
                else:
                    ingr.append(span.text.strip())
        
        return ingr
    
    def GETtitle(self, html_content):
        
        soup = BeautifulSoup(html_content, 'html.parser')

        ### bl class w mn gherr class

        
        titleH1= soup.find('h1', {'class': 'recipe-title'})
        
        ### hena mn gheer 

        if not soup.find('h1', {'class': 'recipe-title'}):
            
            titleH1 = soup.find('h1')
            return titleH1.text.strip()
        else:
            return "Unknown Recipe"
    def extract_instructions(self,html_content):
   
        soup = BeautifulSoup(html_content, 'html.parser')
        
        instructions = []
        
        heading = soup.find('h2', {'class': 'mm-recipes-structured-ingredients__heading'})
        
        if heading:

            instructionsItems = []
            
            parent = heading.parent
            if parent:
                instructionsItems = parent.find_all('li', {'class': 'mntl-sc-block-group--LI'})
            
            if not instructionsItems:
                instructionsItems = soup.find_all('li', {'class': 'mntl-sc-block-group--LI'})
            
            for item in instructionsItems:
                ## select paragraphs b class name 
                
                paragraphs = item.find_all('p', {'class': 'mntl-sc-block-html'})
                
                for p in paragraphs:


                    ## hlf 3la kol paragraph w a3ml extract ll text ely mwgoda
                
                    instructionsText = p.text.strip()
                    if instructionsText and not instructionsText.startswith('Gather all ingredients'):
                        instructions.append(instructionsText)
        # lw ana ml2stsh ay haga s3tha habos 3la class ingredients-item
        ### deh kanet case mwgoda fi pages 3la all recipes

        if not instructions:
          
            for item in soup.find_all('li', {'class': 'ingredients-item'}):

                instructionsText = item.text.strip()
                if item.text.strip():
                    instructions.append(item.text.strip())
        
        return instructions
    
    
def googleSearch(query, num_results=10):
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        
        site_restrictions =  "site:allrecipes.com"
        cook = "site:allrecipes.com"
 
        
        result = service.cse().list(q=f"{query} {cook} {site_restrictions}", cx=SEARCH_ENGINE_ID, num=num_results).execute()
        
        if "items" not in result:
            return []
        
        search_results = []
        for item in result["items"]:
              
            ##### apending results

            #### 
            #### 
            url = item["link"]
            title = item["title"]
        
            search_results.append({"url": url, "title": title})
        
        return search_results
    
    except:
        return []
  
def get_recipes_for_query(query, max_results=2):
    search_results = googleSearch(query, max_results * 2)
    # print(f"Total search results: {len(search_results)}")
    # print(f"Search results: {search_results}")

    if not search_results:
        return []
    
    extractor = RecipeExtractor()
    recipes = []
    
    for result in search_results[:max_results]:
        time.sleep(1)
        
        recipe = extractor.extractUrl(result["url"])
        
        if recipe and "error" not in recipe:
            if recipe["ingredients"] and recipe["instructions"]:
                recipes.append(recipe)
    
    return recipes

def GetRecipe(recipe):
    ### HENA hageeb kol haga mn title w ingredients w instructions

    title = recipe.get('title', 'this dish')

    ######### ingredients w instructions
    ingredients = recipe.get('ingredients', [])
    instructions = recipe.get('instructions', [])
    
    ###### a3mal format response based 3la data ely gbtha
    res = f"Hey! So I wanted to tell you about this amazing {title}. It's super tasty and pretty easy to make.\n\n"
    
    res += "For this recipe, you'll need to grab:\n"
    
    
    ###### b3d kda hlf 3la kol ingredient w a3ml format 7lwa
    for ingredient in ingredients:
        res += f"- {ingredient}\n"
    
    res += f"\nOkay, so here's how to make the {title}:\n\n"
    
    inst = ""
    for i, instruction in enumerate(instructions):

        ######## concatenat instructions fi response 3ndy
        ### bs bshouf lw da awl instruction aw akher wa7ed
        ### w a3ml format 3lehom
        if i == 0:
            inst += f"First, {instruction.lower()} "
        elif i == len(instructions) - 1:
            inst += f"Finally, {instruction.lower()} "
        else:
            l = ["Then", "After that", "Next", "Once that's done"]
            ll = l[i % len(l)]
            inst += f"{ll}, {instruction.lower()} "
    
    res += inst.strip() + "\n\n"
    
    res += f"The {title} is ready! I like to pair it with a simple side salad or some kimchi if you have it. The flavor is so good - a perfect balance of sweet and savory with just a hint of spice. Let me know how it turns out if you try it!"
    
    return res