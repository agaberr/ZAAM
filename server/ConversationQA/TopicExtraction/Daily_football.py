import requests
from bs4 import BeautifulSoup
import csv
import time
import os
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin
import json

class ImprovedFootballScraper:
    def __init__(self, delay=2):
        self.base_url = "https://www.espn.com"
        self.delay = delay
        self.scrape_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create a session for persistent connections
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        })
    
    def make_request(self, url):
        try:
            print(f"Requesting: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error requesting {url}: {e}")
            return None
        finally:
            # Rate limiting
            time.sleep(self.delay)
    
    def scrape_match_results(self, date=None):
        results = []
        
        # Format date parameter
        if date is None:
            # Use today's date if none provided
            date = datetime.now().strftime('%Y%m%d')
        
        # ESPN scores page for soccer with date parameter
        scores_url = f"https://www.espn.com/soccer/scoreboard/_/date/{date}"
        print(f"Fetching scores from: {scores_url}")
        
        response = self.make_request(scores_url)
        if not response:
            return results
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all match sections
        match_sections = soup.select("section.column-content")
        print(f"Found {len(match_sections)} match sections")
        
        # Process each section which might contain multiple matches
        for section in match_sections:
            # Find the competition/league header for this section
            competition_header = section.find_previous("header", class_="Card__Header")
            competition_name = None
            
            if competition_header:
                title_element = competition_header.select_one(".Card__Header__Title")
                if title_element:
                    competition_name = title_element.get_text(strip=True)
            
            # Find all match containers in this section
            match_containers = section.select(".ScoreboardScoreCell")
            print(f"Found {len(match_containers)} matches in section")
            
            for match in match_containers:
                match_data = {}
                
                # Add competition name if found
                if competition_name:
                    match_data['competition'] = competition_name
                
                # Determine if match is upcoming (pre-match)
                match_classes = match.get('class', [])
                is_upcoming = 'ScoreboardScoreCell--pre' in match_classes
                match_data['is_upcoming'] = is_upcoming
                
                # Get match time or status
                # Try the exact selector from the example
                time_element = match.select_one(".ScoreCell__Time.ScoreboardScoreCell__Time")
                
                # If not found, try alternative selectors
                if not time_element:
                    time_element = match.select_one(".ScoreCell__Time")
                
                if time_element:
                    time_text = time_element.get_text(strip=True)
                    if is_upcoming:
                        match_data['scheduled_time'] = time_text
                    else:
                        match_data['status'] = time_text
                else:
                    # Set default values if no time element found
                    if is_upcoming:
                        match_data['scheduled_time'] = ""
                    else:
                        match_data['status'] = ""
                
                # Get teams and scores/records
                home_item = match.select_one(".ScoreboardScoreCell__Item--home")
                away_item = match.select_one(".ScoreboardScoreCell__Item--away")
                
                if home_item and away_item:
                    # Get team names
                    home_team_element = home_item.select_one(".ScoreCell__TeamName")
                    away_team_element = away_item.select_one(".ScoreCell__TeamName")
                    
                    if home_team_element and away_team_element:
                        match_data['home_team'] = home_team_element.get_text(strip=True)
                        match_data['away_team'] = away_team_element.get_text(strip=True)
                    
                    # Get team records
                    home_record_element = home_item.select_one(".ScoreboardScoreCell__Record")
                    away_record_element = away_item.select_one(".ScoreboardScoreCell__Record")
                    
                    if home_record_element and away_record_element:
                        match_data['home_record'] = home_record_element.get_text(strip=True)
                        match_data['away_record'] = away_record_element.get_text(strip=True)
                    
                    # Get scores if match is not upcoming
                    if not is_upcoming:
                        home_score_element = home_item.select_one(".ScoreCell__Score")
                        away_score_element = away_item.select_one(".ScoreCell__Score")
                        
                        if home_score_element and away_score_element:
                            match_data['home_score'] = home_score_element.get_text(strip=True)
                            match_data['away_score'] = away_score_element.get_text(strip=True)
                    
                    # Get team URLs
                    home_team_link = home_item.select_one("a.AnchorLink")
                    away_team_link = away_item.select_one("a.AnchorLink")
                    
                    if home_team_link and home_team_link.has_attr('href'):
                        match_data['home_team_url'] = urljoin(self.base_url, home_team_link['href'])
                    
                    if away_team_link and away_team_link.has_attr('href'):
                        match_data['away_team_url'] = urljoin(self.base_url, away_team_link['href'])
                
                # Find match URL - check for game summary link
                match_link = match.find_parent("a", class_="AnchorLink")
                if match_link and match_link.has_attr('href'):
                    match_data['match_url'] = urljoin(self.base_url, match_link['href'])
                
                # Get broadcast network info if available (especially for upcoming matches)
                network_element = match.select_one(".ScoreCell__Network")
                if network_element:
                    network_items = network_element.select(".ScoreCell__NetworkItem")
                    if network_items:
                        match_data['broadcast'] = [item.get_text(strip=True) for item in network_items]
                
                # If we have basic match data, add to results
                if match_data and 'home_team' in match_data and 'away_team' in match_data:
                    results.append(match_data)
        
        # If no results found with section-based approach, try direct scraping
        if not results:
            print("Using alternative direct scraping approach")
            
            # Group matches by competition
            competition_cards = soup.select(".Card")
            for card in competition_cards:
                competition_header = card.select_one("header.Card__Header")
                competition_name = None
                
                if competition_header:
                    title_element = competition_header.select_one(".Card__Header__Title")
                    if title_element:
                        competition_name = title_element.get_text(strip=True)
                
                # Find matches in this competition card
                match_containers = card.select(".ScoreboardScoreCell")
                
                for match in match_containers:
                    match_data = {}
                    
                    # Add competition name if found
                    if competition_name:
                        match_data['competition'] = competition_name
                    
                    # Determine if match is upcoming (pre-match)
                    match_classes = match.get('class', [])
                    is_upcoming = 'ScoreboardScoreCell--pre' in match_classes
                    match_data['is_upcoming'] = is_upcoming
                    
                    # Get match time or status
                    # Try the exact selector from the example
                    time_element = match.select_one(".ScoreboardScoreCellTime")
                    print("debug: time_element: ",time_element)
                    
                    # If not found, try alternative selectors
                    if not time_element:
                        time_element = match.select_one(".ScoreCell__Time")
                    
                    if time_element:
                        time_text = time_element.get_text(strip=True)
                        if is_upcoming:
                            match_data['scheduled_time'] = time_text
                        else:
                            match_data['status'] = time_text
                    else:
                        # Set default values if no time element found
                        if is_upcoming:
                            match_data['scheduled_time'] = ""
                        else:
                            match_data['status'] = ""
                    
                    # Get teams and scores
                    home_item = match.select_one(".ScoreboardScoreCell__Item--home")
                    away_item = match.select_one(".ScoreboardScoreCell__Item--away")
                    
                    if home_item and away_item:
                        # Extract team names
                        home_team_element = home_item.select_one(".ScoreCell__TeamName")
                        away_team_element = away_item.select_one(".ScoreCell__TeamName")
                        
                        if home_team_element and away_team_element:
                            match_data['home_team'] = home_team_element.get_text(strip=True)
                            match_data['away_team'] = away_team_element.get_text(strip=True)
                        
                        # Get team records
                        home_record_element = home_item.select_one(".ScoreboardScoreCell__Record")
                        away_record_element = away_item.select_one(".ScoreboardScoreCell__Record")
                        
                        if home_record_element and away_record_element:
                            match_data['home_record'] = home_record_element.get_text(strip=True)
                            match_data['away_record'] = away_record_element.get_text(strip=True)
                        
                        # Get scores if match is not upcoming
                        if not is_upcoming:
                            home_score_element = home_item.select_one(".ScoreCell__Score")
                            away_score_element = away_item.select_one(".ScoreCell__Score")
                            
                            if home_score_element and away_score_element:
                                match_data['home_score'] = home_score_element.get_text(strip=True)
                                match_data['away_score'] = away_score_element.get_text(strip=True)
                    
                    # Get broadcast network info
                    network_element = match.select_one(".ScoreCell__Network")
                    if network_element:
                        network_items = network_element.select(".ScoreCell__NetworkItem")
                        if network_items:
                            match_data['broadcast'] = [item.get_text(strip=True) for item in network_items]
                    
                    if match_data and 'home_team' in match_data and 'away_team' in match_data:
                        results.append(match_data)
        
        print(f"Total matches found: {len(results)}")
        return results
    
    def scrape_league_tables(self):
        tables = {}

        leagues = [
            {"name": "Premier League", "url": "https://global.espn.com/football/table/_/league/eng.1"},
            {"name": "La Liga", "url": "https://global.espn.com/football/table/_/league/esp.1"},
            {"name": "Serie A", "url": "https://global.espn.com/football/table/_/league/ita.1"},
            {"name": "Bundesliga", "url": "https://global.espn.com/football/table/_/league/ger.1"},
            {"name": "Ligue 1", "url": "https://global.espn.com/football/table/_/league/fra.1"}
        ]

        for league in leagues:
            league_table = []
            print(f"Scraping: {league['name']}")
            
            response = self.make_request(league["url"])
            if not response:
                print(f"Failed to get response for {league['name']}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the responsive tables (ESPN's current layout)
            responsive_tables = soup.select(".ResponsiveTable")
            
            if responsive_tables:
                for resp_table in responsive_tables:
                    # Check ESPN's split table structure (team names and stats)
                    left_table = resp_table.select_one("table.Table--fixed-left")
                    right_scroller = resp_table.select_one(".Table__ScrollerWrapper")
                    
                    if not left_table or not right_scroller:
                        continue
                    
                    right_table = right_scroller.select_one("table")
                    if not right_table:
                        continue
                    
                    # Get team rows from left table
                    team_rows = left_table.select("tbody tr")
                    # Get stats rows from right table
                    stats_rows = right_table.select("tbody tr")
                    
                    if len(team_rows) != len(stats_rows):
                        print(f"Mismatch in row counts for {league['name']}")
                        continue
                    
                    for i, (team_row, stats_row) in enumerate(zip(team_rows, stats_rows)):
                        team_data = {}
                        
                        # Extract position
                        position_elem = team_row.select_one(".team-position")
                        if position_elem:
                            team_data["position"] = position_elem.get_text(strip=True)
                        else:
                            team_data["position"] = str(i + 1)
                        
                        # Extract team name
                        team_name_elem = team_row.select_one(".hide-mobile a")
                        if not team_name_elem:
                            team_name_elem = team_row.select_one("a")
                        
                        if team_name_elem:
                            team_data["team"] = team_name_elem.get_text(strip=True)
                        else:
                            continue  # Skip if we can't find the team name
                        
                        # Extract stats
                        stat_cells = stats_row.select("td span.stat-cell")
                        if len(stat_cells) >= 8:  # Make sure we have all stats
                            team_data["played"] = stat_cells[0].get_text(strip=True)
                            team_data["won"] = stat_cells[1].get_text(strip=True)
                            team_data["drawn"] = stat_cells[2].get_text(strip=True)
                            team_data["lost"] = stat_cells[3].get_text(strip=True)
                            team_data["goals_for"] = stat_cells[4].get_text(strip=True)
                            team_data["goals_against"] = stat_cells[5].get_text(strip=True)
                            team_data["goal_diff"] = stat_cells[6].get_text(strip=True)
                            team_data["points"] = stat_cells[7].get_text(strip=True)
                        
                            league_table.append(team_data)
            
            tables[league["name"]] = league_table
            print(f"Scraped {len(league_table)} teams for {league['name']}")
        
        return tables
    
    def scrape_football_data(self):
   
        print("Starting comprehensive football data scraping...")
        data = {
            'scrape_date': self.scrape_date,
            'match_results': [],
            'upcoming_matches': [],
            'team_news': [],
            'league_tables': {}
        }
        
        # Scrape match results
        print("Scraping recent match results...")
        results = self.scrape_match_results(20250423)
        data['match_results'] = results

        data['league_tables'] = self.scrape_league_tables()
        
        return data
    
    
    def format_team_news(self, news):

        if not news:
            return "No team news available."
            
        text = "=== TEAM NEWS ===\n\n"
        
        # Group by team and deduplicate articles by title
        news_by_team = {}
        seen_titles = set()
        
        for article in news:
            team = article.get('team', 'Unknown')
            title = article.get('title', '')
            
            # Skip duplicate articles
            if title in seen_titles:
                continue
                
            if title:
                seen_titles.add(title)
                
            if team not in news_by_team:
                news_by_team[team] = []
            news_by_team[team].append(article)
        
        for team, team_articles in news_by_team.items():
            text += f"--- {team} ---\n\n"
            
            for article in team_articles:
                text += f"{article.get('title', 'Unknown')}\n\n"
                if 'preview' in article and article['preview']:
                    text += f"{article['preview']}\n\n"
                if 'url' in article and article['url']:
                    text += f"Read more: {article['url']}\n\n"
        
        return text

    
    def format_league_tables(self, tables):
        if not tables:
            return "No league tables available."
            
        text = "=== LEAGUE STANDINGS ===\n\n"
        
        for league_name, table in tables.items():
            if not table:
                continue
                
            text += f"--- {league_name} ---\n"
            
            # Create formatted table header
            text += f"{'Pos':<4}{'Team':<25}{'GP':<4}{'W':<4}{'D':<4}{'L':<4}{'GF':<4}{'GA':<4}{'GD':<4}{'Pts':<4}\n"
            
            # Add separator line
            text += "-" * 65 + "\n"
            
            # Format each team's data
            for team in table:
                text += f"{team.get('position', ''):<4}"
                text += f"{team.get('team', ''):<25}"
                text += f"{team.get('games_played', ''):<4}"
                text += f"{team.get('wins', ''):<4}"
                text += f"{team.get('draws', ''):<4}"
                text += f"{team.get('losses', ''):<4}"
                text += f"{team.get('goals_for', ''):<4}"
                text += f"{team.get('goals_against', ''):<4}"
                text += f"{team.get('goal_diff', ''):<4}"
                text += f"{team.get('points', ''):<4}\n"
            
            text += "\n"
        
        return text



# Example usage
if __name__ == "__main__":
    scraper = ImprovedFootballScraper(delay=3)
    
    # Scrape all football data
    football_data = scraper.scrape_football_data()
    
    # # Save data to formatted text file
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # scraper.save_football_data(football_data, f"football_report_{timestamp}.txt")
    print(football_data)
    
    print("Football data scraping completed successfully!")