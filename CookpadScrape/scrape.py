import requests
from bs4 import BeautifulSoup
import json
import time
import csv
import re
from urllib.parse import urljoin, quote
import random

class MassRecipeScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        self.delay_min = 1
        self.delay_max = 3
        self.all_recipes = []
    
    def random_delay(self):
        """Random delay untuk avoid detection"""
        delay = random.uniform(self.delay_min, self.delay_max)
        time.sleep(delay)
    
    def get_page(self, url):
        try:
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            self.random_delay()
            return response
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

class CookpadMassScraper(MassRecipeScraper):
    def __init__(self):
        super().__init__()
        self.base_url = "https://cookpad.com"
    
    def extract_cookpad_recipe(self, soup, url):
        """Extract recipe dari Cookpad dengan lebih detail"""
        try:
            # Title
            title_elem = soup.find('h1', class_='recipe-title')
            if not title_elem:
                title_elem = soup.find('h1')
                if not title_elem:
                    title_elem = soup.find('title')
            
            title = title_elem.get_text().strip() if title_elem else "Unknown Recipe"
            title = re.sub(r'\s+', ' ', title).strip()
            
            # Ingredients
            ingredients = []
            
            # Try multiple selectors for ingredients
            ingredient_selectors = [
                'li.ingredient',
                'div.ingredient',
                '.recipe-ingredient-list li',
                '.ingredients-section li',
                '[class*="ingredient"] li'
            ]
            
            for selector in ingredient_selectors:
                items = soup.select(selector)
                if items:
                    for item in items:
                        text = item.get_text().strip()
                        text = re.sub(r'\s+', ' ', text)
                        if text and len(text) > 2:
                            ingredients.append(text)
                    break
            
            # If still no ingredients, try broader search
            if not ingredients:
                potential_ingredients = soup.find_all(text=re.compile(r'\d+\s*(gr|ml|sdm|sdt|buah|siung|lembar)'))
                for text in potential_ingredients[:10]:  # Limit to 10
                    clean_text = text.strip()
                    if clean_text:
                        ingredients.append(clean_text)
            
            # Steps/Instructions
            instructions = []
            
            # Try multiple selectors for steps
            step_selectors = [
                'li.step',
                'div.step',
                '.recipe-instruction-list li',
                '.instructions-section li',
                '[class*="step"] li',
                '.step-content'
            ]
            
            for selector in step_selectors:
                items = soup.select(selector)
                if items:
                    for i, item in enumerate(items, 1):
                        text = item.get_text().strip()
                        text = re.sub(r'\s+', ' ', text)
                        if text and len(text) > 10:
                            instructions.append(f"{i}) {text}")
                    break
            
            # If no structured steps, try to extract from paragraphs
            if not instructions:
                paragraphs = soup.find_all('p')
                step_count = 1
                for p in paragraphs:
                    text = p.get_text().strip()
                    if len(text) > 20 and any(word in text.lower() for word in ['masak', 'goreng', 'rebus', 'tumis', 'campur']):
                        instructions.append(f"{step_count}) {text}")
                        step_count += 1
                        if len(instructions) >= 5:  # Limit to 5 steps
                            break
            
            # Clean title
            title = title.replace('Resep', '').replace('resep', '').strip()
            
            if ingredients and instructions and title != "Unknown Recipe":
                return {
                    'title': title,
                    'ingredients': ' , '.join(ingredients),
                    'steps': '\n'.join(instructions),
                    'url': url
                }
            
            return None
            
        except Exception as e:
            print(f"Error extracting Cookpad recipe: {e}")
            return None
    
    def search_cookpad_recipes(self, query, max_pages=20):
        """Search recipes di Cookpad dengan multiple pages"""
        recipe_urls = []
        
        for page in range(1, max_pages + 1):
            search_url = f"https://cookpad.com/id/cari/{quote(query)}?page={page}"
            
            response = self.get_page(search_url)
            if not response:
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find recipe links
            links = soup.find_all('a', href=re.compile(r'/id/resep/\d+'))
            
            page_urls = []
            for link in links:
                href = link.get('href')
                if href and '/resep/' in href:
                    full_url = urljoin(self.base_url, href)
                    if full_url not in recipe_urls:
                        recipe_urls.append(full_url)
                        page_urls.append(full_url)
            
            print(f"Page {page}: Found {len(page_urls)} new recipe URLs")
            
            # If no new URLs found, probably reached end
            if not page_urls:
                print(f"No more recipes found at page {page}, stopping search")
                break
        
        return recipe_urls
    
    def scrape_cookpad_mass(self, search_queries, target_recipes=1000):
        """Scrape banyak resep dari Cookpad"""
        
        # Search queries yang akan digunakan
        default_queries = [
            "ayam goreng", "nasi goreng", "rendang", "soto ayam", "gado gado",
            "bakso", "mie ayam", "gudeg", "sambal", "tumis kangkung",
            "ikan bakar", "sayur asem", "opor ayam", "pecel lele", "rawon",
            "sate ayam", "capcay", "ayam bakar", "sup ayam", "telur dadar",
            "tempe goreng", "tahu goreng", "perkedel", "pizza", "pasta",
            "chicken curry", "beef steak", "fried rice", "soup", "salad"
        ]
        
        if not search_queries:
            search_queries = default_queries
        
        all_recipe_urls = []
        
        for query in search_queries:
            print(f"\n=== Searching for '{query}' recipes ===")
            
            # Calculate how many pages needed per query
            recipes_per_query = target_recipes // len(search_queries)
            pages_needed = max(1, recipes_per_query // 20)  # ~20 recipes per page
            
            query_urls = self.search_cookpad_recipes(query, max_pages=pages_needed)
            all_recipe_urls.extend(query_urls)
            
            print(f"Total URLs collected so far: {len(all_recipe_urls)}")
            
            # Stop if we have enough URLs
            if len(all_recipe_urls) >= target_recipes:
                break
        
        # Remove duplicates and limit to target
        unique_urls = list(set(all_recipe_urls))[:target_recipes]
        print(f"\n🎯 Ready to scrape {len(unique_urls)} unique recipe URLs")
        
        # Scrape all recipes
        successful_recipes = []
        failed_count = 0
        
        for i, url in enumerate(unique_urls, 1):
            print(f"\nScraping recipe {i}/{len(unique_urls)} ({len(successful_recipes)} successful)")
            
            response = self.get_page(url)
            if not response:
                failed_count += 1
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            recipe_data = self.extract_cookpad_recipe(soup, url)
            
            if recipe_data:
                successful_recipes.append(recipe_data)
                print(f"✓ {recipe_data['title']}")
            else:
                failed_count += 1
                print(f"✗ Failed to extract recipe data")
            
            # Progress update setiap 50 recipes
            if i % 50 == 0:
                print(f"\n📊 Progress: {i}/{len(unique_urls)} processed, {len(successful_recipes)} successful, {failed_count} failed")
        
        return successful_recipes

class TastyScraper(MassRecipeScraper):
    """Scraper untuk Tasty.co - website yang lebih permisif"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://tasty.co"
    
    def search_tasty_recipes(self, max_pages=30):
        """Scrape dari Tasty.co recipe pages"""
        recipe_urls = []
        
        # Different category URLs
        categories = [
            "https://tasty.co/topic/easy",
            "https://tasty.co/topic/healthy",
            "https://tasty.co/topic/chicken",
            "https://tasty.co/topic/pasta",
            "https://tasty.co/topic/desserts",
            "https://tasty.co/topic/vegetarian",
            "https://tasty.co/topic/quick",
            "https://tasty.co/topic/comfort-food"
        ]
        
        for category_url in categories:
            print(f"Scraping category: {category_url}")
            
            for page in range(1, 5):  # 4 pages per category
                page_url = f"{category_url}?page={page}"
                
                response = self.get_page(page_url)
                if not response:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find recipe links
                links = soup.find_all('a', href=re.compile(r'/recipe/'))
                
                for link in links:
                    href = link.get('href')
                    if href and '/recipe/' in href:
                        full_url = urljoin(self.base_url, href)
                        if full_url not in recipe_urls:
                            recipe_urls.append(full_url)
        
        return recipe_urls[:500]  # Limit to 500 from Tasty
    
    def extract_tasty_recipe(self, soup, url):
        """Extract recipe dari Tasty.co"""
        try:
            # Try JSON-LD first
            scripts = soup.find_all('script', type='application/ld+json')
            
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, list):
                        for item in data:
                            if item.get('@type') == 'Recipe':
                                return self.parse_tasty_json(item, url)
                    elif data.get('@type') == 'Recipe':
                        return self.parse_tasty_json(data, url)
                except:
                    continue
            
            # Fallback to HTML parsing
            return self.extract_tasty_html(soup, url)
            
        except Exception as e:
            print(f"Error extracting Tasty recipe: {e}")
            return None
    
    def parse_tasty_json(self, recipe_data, url):
        """Parse Tasty recipe dari JSON-LD"""
        try:
            name = recipe_data.get('name', '')
            
            ingredients = []
            for ingredient in recipe_data.get('recipeIngredient', []):
                if isinstance(ingredient, str):
                    ingredients.append(ingredient.strip())
            
            instructions = []
            for i, step in enumerate(recipe_data.get('recipeInstructions', []), 1):
                if isinstance(step, dict):
                    text = step.get('text', '')
                else:
                    text = str(step)
                
                if text:
                    instructions.append(f"{i}) {text.strip()}")
            
            if name and ingredients and instructions:
                return {
                    'title': name,
                    'ingredients': ' , '.join(ingredients),
                    'steps': '\n'.join(instructions),
                    'url': url
                }
            
            return None
            
        except Exception as e:
            print(f"Error parsing Tasty JSON: {e}")
            return None

# Main execution untuk mass scraping
if __name__ == "__main__":
    print("🚀 MASS RECIPE SCRAPER - TARGET: 1000+ RECIPES 🚀")
    
    # Custom search queries (bisa diubah sesuai keinginan)
    custom_queries = [
        # Indonesian dishes
        "ayam goreng", "nasi goreng", "rendang", "soto ayam", "gado gado",
        "bakso", "mie ayam", "gudeg", "sambal", "tumis kangkung",
        "ikan bakar", "sayur asem", "opor ayam", "pecel lele", "rawon",
        "sate ayam", "capcay", "ayam bakar", "sup ayam", "telur dadar",
        "tempe goreng", "tahu goreng", "perkedel", "ayam rica rica",
        "nasi padang", "martabak", "pempek", "kerak telor", "asinan",
        
        # International dishes  
        "chicken curry", "beef steak", "fried rice", "pasta", "pizza",
        "sandwich", "burger", "salad", "soup", "noodles",
        "fish fillet", "roast chicken", "grilled salmon", "beef stir fry",
        "vegetable curry", "fried chicken", "rice bowl", "seafood pasta"
    ]
    
    # Initialize Cookpad scraper
    cookpad_scraper = CookpadMassScraper()
    
    print(f"🎯 Target: 1000 recipes")
    print(f"📝 Using {len(custom_queries)} search queries")
    print(f"🔍 Search queries: {', '.join(custom_queries[:10])}...")
    
    # Start mass scraping
    print("\n" + "="*60)
    print("STARTING MASS SCRAPING FROM COOKPAD")
    print("="*60)
    
    all_recipes = cookpad_scraper.scrape_cookpad_mass(
        search_queries=custom_queries,
        target_recipes=1000
    )
    
    # Save results
    if all_recipes:
        print(f"\n🎉 SUCCESS! Scraped {len(all_recipes)} recipes!")
        
        # Save to CSV format yang diminta
        with open('mass_recipes_1000.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Title Cleaned', 'Ingredients Cleaned', 'Steps']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for recipe in all_recipes:
                writer.writerow({
                    'Title Cleaned': recipe['title'],
                    'Ingredients Cleaned': recipe['ingredients'],
                    'Steps': recipe['steps']
                })
        
        # Save to JSON
        formatted_recipes = []
        for recipe in all_recipes:
            formatted_recipes.append({
                'Title Cleaned': recipe['title'],
                'Ingredients Cleaned': recipe['ingredients'],
                'Steps': recipe['steps']
            })
        
        with open('mass_recipes_1000.json', 'w', encoding='utf-8') as jsonfile:
            json.dump(formatted_recipes, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Files saved:")
        print(f"   - mass_recipes_1000.csv ({len(all_recipes)} recipes)")
        print(f"   - mass_recipes_1000.json ({len(all_recipes)} recipes)")
        
        # Statistics
        print(f"\n📊 STATISTICS:")
        print(f"   - Total recipes scraped: {len(all_recipes)}")
        print(f"   - Average ingredients per recipe: {sum(len(r['ingredients'].split(',')) for r in all_recipes) / len(all_recipes):.1f}")
        # print(f"   - Average steps per recipe: {sum(len(r['steps'].split('\\n')) for r in all_recipes) / len(all_recipes):.1f}")
        
    else:
        print("❌ No recipes were scraped successfully.")
    
    print("\n✨ SCRAPING COMPLETED! ✨")