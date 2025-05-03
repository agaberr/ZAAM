import json
import re
from typing import Dict, List, Any, Tuple, Optional
import string
from collections import Counter

class FootballQASystem:
    def __init__(self, data_file: str = None, data_dict: dict = None):
        if data_dict:
            self.data = data_dict
        elif data_file:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            raise ValueError("Either data_file or data_dict must be provided")
        
        # Extract key information for faster lookup
        self.matches = self.data.get("match_results", [])
        self.league_tables = self.data.get("league_tables", {})
        self.upcoming_matches = self.data.get("match_results", [])
        self.team_news = self.data.get("team_news", [])
        
        # Create lookup dictionaries for faster access
        self.teams_info = self._extract_teams_info()
        self.competitions = self._extract_competitions()
        
        # Build a simple keyword dictionary for matching questions to data
        self.query_templates = self._build_query_templates()
    
    def _extract_teams_info(self) -> Dict[str, Dict]:
        """Extract team information from league tables and matches for faster lookup"""
        teams_info = {}
        
        # Extract from league tables
        for league, table in self.league_tables.items():
            for entry in table:
                team_name = entry.get("team")
                if team_name:
                    if team_name not in teams_info:
                        teams_info[team_name] = {"leagues": [league], "positions": {league: entry}}
                    else:
                        teams_info[team_name]["leagues"].append(league)
                        teams_info[team_name]["positions"][league] = entry
        
        # Add match data
        for match in self.matches:
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            
            for team in [home_team, away_team]:
                if team and team not in teams_info:
                    teams_info[team] = {"matches": [match]}
                elif team:
                    if "matches" not in teams_info[team]:
                        teams_info[team]["matches"] = [match]
                    else:
                        teams_info[team]["matches"].append(match)
        
        return teams_info
    
    def _extract_competitions(self) -> Dict[str, Dict]:
        """Extract competition information for faster lookup"""
        competitions = {}
        
        # Extract from league tables
        for league_name, table in self.league_tables.items():
            competitions[league_name] = {"table": table}
        
        # Group matches by competition
        for match in self.matches:
            comp = match.get("competition")
            if comp:
                if comp not in competitions:
                    competitions[comp] = {"matches": [match]}
                else:
                    if "matches" not in competitions[comp]:
                        competitions[comp]["matches"] = [match]
                    else:
                        competitions[comp]["matches"].append(match)
        
        return competitions
    
    def _build_query_templates(self) -> Dict[str, List[str]]:
        templates = {
            "league_position": ["position", "standing", "rank", "table", "placed", "league", "standings"],
            "team_stats": ["goals", "scored", "conceded", "wins", "losses", "draws", "points", "record"],
            "match_result": ["score", "result", "win", "lose", "draw", "match", "game", "played", "defeat", "victory"],
            "team_comparison": ["better", "worse", "compare", "versus", "vs", "against", "stronger", "weaker"],
            "league_leaders": ["top", "leader", "champion", "winning", "best", "first", "highest"],
            "relegation": ["bottom", "relegation", "worst", "last", "lowest"],
            "upcoming_matches": ["upcoming", "next", "schedule", "fixture", "when", "play next"],
            "todays_matches": ["today", "tonight", "this evening", "happening now"]
        }
        return templates
    
    def _find_team_mentions(self, question: str) -> List[str]:
        """Find team names mentioned in the question"""
        found_teams = []
        question_lower = question.lower()
        
        for team_name in self.teams_info.keys():
            # Create pattern for team name that handles special characters
            team_pattern = re.escape(team_name.lower())
            
            if re.search(r'\b' + team_pattern + r'\b', question_lower):
                found_teams.append(team_name)
        
        return found_teams
    
    def _find_league_mentions(self, question: str) -> List[str]:
        """Find league names mentioned in the question"""
        found_leagues = []
        question_lower = question.lower()
        
        for league_name in self.league_tables.keys():
            league_pattern = re.escape(league_name.lower())
            
            if re.search(r'\b' + league_pattern + r'\b', question_lower):
                found_leagues.append(league_name)
        
        # Handle common league name variations
        common_leagues = {
            "epl": "Premier League",
            "premier league": "Premier League",
            "la liga": "La Liga",
            "serie a": "Serie A",
            "bundesliga": "Bundesliga",
            "ligue 1": "Ligue 1"
        }
        
        for variant, formal_name in common_leagues.items():
            if variant in question_lower and formal_name in self.league_tables:
                if formal_name not in found_leagues:
                    found_leagues.append(formal_name)
        
        return found_leagues
    
    def _determine_question_type(self, question: str) -> Tuple[str, float]:
        """Determine the type of question based on keywords"""
        question_lower = question.lower()
        scores = {}
        
        for q_type, keywords in self.query_templates.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                scores[q_type] = score
        
        if not scores:
            return "general", 0
        
        best_match = max(scores.items(), key=lambda x: x[1])
        return best_match[0], best_match[1]
    
    def answer_question(self, question: str) -> str:
        # Step 1: Find mentioned teams and leagues
        teams = self._find_team_mentions(question)
        leagues = self._find_league_mentions(question)
        
        # Step 2: Determine question type
        question_type, confidence = self._determine_question_type(question)
        
        # Step 3: Generate answer based on question type and entities
        answer = self._generate_answer(question, question_type, teams, leagues)
        
        # Step 4: If no good answer, give a generic response
        if not answer:
            return self._generate_fallback_response(question)
        
        return answer
    
    def _generate_answer(self, question: str, question_type: str, teams: List[str], leagues: List[str]) -> Optional[str]:
        """Generate an answer based on question type and mentioned entities"""
        question_lower = question.lower()
        
        # Handle league position questions
        if question_type == "league_position" and teams:
            return self._answer_league_position(question, teams, leagues)
        
        # Handle team stats questions
        elif question_type == "team_stats" and teams:
            return self._answer_team_stats(question, teams, leagues)
        
        # Handle match result questions
        elif question_type == "match_result" and (len(teams) >= 1):
            return self._answer_match_result(question, teams)
        
        # Handle team comparison questions
        elif question_type == "team_comparison" and len(teams) >= 2:
            return self._answer_team_comparison(question, teams, leagues)
        
        # Handle league leaders questions
        elif question_type == "league_leaders" and leagues:
            return self._answer_league_leaders(question, leagues)
        
        # Handle relegation questions
        elif question_type == "relegation" and leagues:
            return self._answer_relegation(question, leagues)
        
        # Handle upcoming matches questions
        elif question_type == "upcoming_matches":
            return self._handle_upcoming_matches(question, teams, leagues)
        
        # Handle today's matches questions
        elif question_type == "todays_matches":
            return self._handle_today_matches(question)
        
        # Generic questions about mentioned teams
        elif teams:
            return self._answer_general_team_question(question, teams)
        
        # Generic questions about mentioned leagues
        elif leagues:
            return self._answer_general_league_question(question, leagues)
        
        if "upcoming" in question_lower or "next" in question_lower or "schedule" in question_lower or "fixture" in question_lower:
            return self._handle_upcoming_matches(question)
        
        # Fallback for general today's matches questions
        if "today" in question_lower or "tonight" in question_lower or "this evening" in question_lower:
            return self._handle_today_matches(question)
        
        return "Sorry, I couldn't find an answer to that question."
    
    def _answer_league_position(self, question: str, teams: List[str], leagues: List[str]) -> str:
        team = teams[0]  # Take the first team mentioned
        
        # If leagues are explicitly mentioned, check those first
        if leagues:
            league = leagues[0]
            if league in self.league_tables:
                for entry in self.league_tables[league]:
                    if entry.get("team") == team:
                        position = entry.get("position")
                        points = entry.get("points")
                        played = entry.get("played")
                        return f"{team} is currently in position {position} in the {league} with {points} points after {played} games."
        
        # If no league is specified or team not found in specified league
        for league_name, table in self.league_tables.items():
            for entry in table:
                if entry.get("team") == team:
                    position = entry.get("position")
                    points = entry.get("points")
                    played = entry.get("played")
                    return f"{team} is currently in position {position} in the {league_name} with {points} points after {played} games."
        
        return f"I couldn't find league position information for {team}."
    
    def _answer_team_stats(self, question: str, teams: List[str], leagues: List[str]) -> str:
        """Answer questions about team statistics"""
        team = teams[0]
        question_lower = question.lower()
        
        # Determine which stat is being asked about
        stat_keywords = {
            "goals scored": ["goals", "scored", "score"],
            "goals conceded": ["conceded", "against", "let in"],
            "wins": ["won", "wins", "victories"],
            "losses": ["lost", "losses", "defeats"],
            "draws": ["draw", "draws", "tied"],
            "points": ["points", "pts"],
            "goal difference": ["goal difference", "goal diff", "gd"],
            "record": ["record", "performance", "form"]
        }
        
        requested_stats = []
        for stat, keywords in stat_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                requested_stats.append(stat)
        
        # If no specific stat is identified, return general stats
        if not requested_stats:
            requested_stats = ["record"]
        
        # If leagues are explicitly mentioned, prioritize those
        target_leagues = leagues if leagues else list(self.league_tables.keys())
        
        for league in target_leagues:
            for entry in self.league_tables.get(league, []):
                if entry.get("team") == team:
                    # Construct response based on requested stats
                    responses = []
                    
                    for stat in requested_stats:
                        if stat == "goals scored":
                            responses.append(f"{team} has scored {entry.get('goals_for')} goals in {entry.get('played')} {league} matches")
                        elif stat == "goals conceded":
                            responses.append(f"{team} has conceded {entry.get('goals_against')} goals in {entry.get('played')} {league} matches")
                        elif stat == "wins":
                            responses.append(f"{team} has won {entry.get('won')} out of {entry.get('played')} {league} matches")
                        elif stat == "losses":
                            responses.append(f"{team} has lost {entry.get('lost')} out of {entry.get('played')} {league} matches")
                        elif stat == "draws":
                            responses.append(f"{team} has drawn {entry.get('drawn')} out of {entry.get('played')} {league} matches")
                        elif stat == "points":
                            responses.append(f"{team} has {entry.get('points')} points in the {league}")
                        elif stat == "goal difference":
                            responses.append(f"{team} has a goal difference of {entry.get('goal_diff')} in the {league}")
                        elif stat == "record":
                            responses.append(f"{team}'s record in the {league} is {entry.get('won')}-{entry.get('drawn')}-{entry.get('lost')} (W-D-L) with {entry.get('points')} points")
                    
                    return " and ".join(responses) + "."
        
        return f"I couldn't find the requested statistics for {team}."
    
    def _answer_match_result(self, question: str, teams: List[str]) -> str:
        """Answer questions about match results"""
        # If one team mentioned, find their latest match
        if len(teams) == 1:
            team = teams[0]
            team_matches = []
            
            for match in self.matches:
                if match.get("home_team") == team or match.get("away_team") == team:
                    team_matches.append(match)
            
            if team_matches:
                # Assume the last match in the list is the most recent
                latest_match = team_matches[-1]
                home_team = latest_match.get("home_team")
                away_team = latest_match.get("away_team")
                home_score = latest_match.get("home_score")
                away_score = latest_match.get("away_score")
                competition = latest_match.get("competition")
                
                result_phrase = ""
                if home_team == team:
                    if int(home_score) > int(away_score):
                        result_phrase = f"won against {away_team}"
                    elif int(home_score) < int(away_score):
                        result_phrase = f"lost to {away_team}"
                    else:
                        result_phrase = f"drew with {away_team}"
                else:
                    if int(away_score) > int(home_score):
                        result_phrase = f"won against {home_team}"
                    elif int(away_score) < int(home_score):
                        result_phrase = f"lost to {home_team}"
                    else:
                        result_phrase = f"drew with {home_team}"
                
                return f"In their most recent match in the {competition}, {team} {result_phrase} with a score of {home_score}-{away_score}."
            
            return f"I couldn't find any recent match results for {team}."
        
        # If two teams mentioned, look for their direct encounter
        elif len(teams) == 2:
            team1, team2 = teams[0], teams[1]
            
            for match in self.matches:
                home_team = match.get("home_team")
                away_team = match.get("away_team")
                
                if (home_team == team1 and away_team == team2) or (home_team == team2 and away_team == team1):
                    home_score = match.get("home_score")
                    away_score = match.get("away_score")
                    competition = match.get("competition")
                    
                    return f"The match between {home_team} and {away_team} in the {competition} ended with a score of {home_score}-{away_score}."
            
            return f"I couldn't find any recent match results between {team1} and {team2}."
        
        return "I need more information about which teams you're asking about."
    
    def _answer_team_comparison(self, question: str, teams: List[str], leagues: List[str]) -> str:
        """Answer questions comparing two teams"""
        if len(teams) < 2:
            return "I need at least two teams to compare."
        
        team1, team2 = teams[0], teams[1]
        
        # If leagues are explicitly mentioned, prioritize those
        target_leagues = leagues if leagues else list(self.league_tables.keys())
        
        for league in target_leagues:
            team1_entry = None
            team2_entry = None
            
            for entry in self.league_tables.get(league, []):
                if entry.get("team") == team1:
                    team1_entry = entry
                elif entry.get("team") == team2:
                    team2_entry = entry
            
            if team1_entry and team2_entry:
                team1_pos = int(team1_entry.get("position"))
                team2_pos = int(team2_entry.get("position"))
                team1_pts = int(team1_entry.get("points"))
                team2_pts = int(team2_entry.get("points"))
                
                comparison = []
                
                # Compare positions
                if team1_pos < team2_pos:
                    comparison.append(f"{team1} is ranked higher than {team2} in the {league} ({team1_pos} vs {team2_pos})")
                elif team1_pos > team2_pos:
                    comparison.append(f"{team2} is ranked higher than {team1} in the {league} ({team2_pos} vs {team1_pos})")
                else:
                    comparison.append(f"{team1} and {team2} are tied in position {team1_pos} in the {league}")
                
                # Compare points
                point_diff = abs(team1_pts - team2_pts)
                if team1_pts > team2_pts:
                    comparison.append(f"{team1} has {point_diff} more points than {team2} ({team1_pts} vs {team2_pts})")
                elif team1_pts < team2_pts:
                    comparison.append(f"{team2} has {point_diff} more points than {team1} ({team2_pts} vs {team1_pts})")
                else:
                    comparison.append(f"Both teams have {team1_pts} points")
                
                return " and ".join(comparison) + "."
        
        # Check if they played against each other
        for match in self.matches:
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            
            if (home_team == team1 and away_team == team2) or (home_team == team2 and away_team == team1):
                home_score = int(match.get("home_score"))
                away_score = int(match.get("away_score"))
                
                if home_team == team1:
                    if home_score > away_score:
                        return f"{team1} defeated {team2} {home_score}-{away_score} in their recent match."
                    elif home_score < away_score:
                        return f"{team2} defeated {team1} {away_score}-{home_score} in their recent match."
                    else:
                        return f"{team1} and {team2} drew {home_score}-{away_score} in their recent match."
                else:
                    if away_score > home_score:
                        return f"{team2} defeated {team1} {away_score}-{home_score} in their recent match."
                    elif away_score < home_score:
                        return f"{team1} defeated {team2} {home_score}-{away_score} in their recent match."
                    else:
                        return f"{team1} and {team2} drew {home_score}-{away_score} in their recent match."
        
        return f"I don't have enough information to compare {team1} and {team2}."
    
    def _answer_league_leaders(self, question: str, leagues: List[str]) -> str:
        """Answer questions about league leaders"""
        league = leagues[0]
        
        if league in self.league_tables and self.league_tables[league]:
            top_teams = self.league_tables[league][:3]  # Get top 3 teams
            
            if top_teams:
                leader = top_teams[0]
                leader_name = leader.get("team")
                leader_points = leader.get("points")
                leader_played = leader.get("played")
                
                response = f"{leader_name} is leading the {league} with {leader_points} points after {leader_played} games."
                
                if len(top_teams) > 1:
                    response += f" They are followed by {', '.join(t.get('team') for t in top_teams[1:])}."
                
                return response
        
        return f"I couldn't find information about the leaders in the {league}."
    
    def _answer_relegation(self, question: str, leagues: List[str]) -> str:
        """Answer questions about teams in relegation zone"""
        league = leagues[0]
        
        if league in self.league_tables and self.league_tables[league]:
            # Usually bottom 3 teams are in relegation zone
            table = self.league_tables[league]
            relegation_teams = table[-3:]
            
            if relegation_teams:
                team_descriptions = []
                for team in relegation_teams:
                    team_descriptions.append(f"{team.get('team')} ({team.get('position')}: {team.get('points')} pts)")
                
                return f"The teams currently in the relegation zone in the {league} are: {', '.join(team_descriptions)}."
        
        return f"I couldn't find information about relegation in the {league}."
    
    def _answer_general_team_question(self, question: str, teams: List[str]) -> str:
        """Answer general questions about a team"""
        team = teams[0]
        team_info = self.teams_info.get(team, {})
        
        if not team_info:
            return f"I don't have much information about {team}."
        
        leagues = team_info.get("leagues", [])
        
        response_parts = []
        
        # Add league position info if available
        for league in leagues:
            if league in team_info.get("positions", {}):
                position = team_info["positions"][league].get("position")
                points = team_info["positions"][league].get("points")
                played = team_info["positions"][league].get("played")
                response_parts.append(f"{team} is currently in position {position} in the {league} with {points} points after {played} games")
        
        # Add recent match info if available
        matches = team_info.get("matches", [])
        if matches:
            recent_match = matches[-1]
            home_team = recent_match.get("home_team")
            away_team = recent_match.get("away_team")
            home_score = recent_match.get("home_score")
            away_score = recent_match.get("away_score")
            
            # Check if we have valid scores before comparing
            if home_score is not None and away_score is not None:
                if home_team == team:
                    opponent = away_team
                    team_score = home_score
                    opp_score = away_score
                    location = "at home"
                else:
                    opponent = home_team
                    team_score = away_score
                    opp_score = home_score
                    location = "away"
                
                result = ""
                try:
                    if int(team_score) > int(opp_score):
                        result = "won"
                    elif int(team_score) < int(opp_score):
                        result = "lost"
                    else:
                        result = "drew"
                    
                    response_parts.append(f"In their most recent match, they {result} {location} against {opponent} with a score of {team_score}-{opp_score}")
                except (TypeError, ValueError):
                    # If we can't convert to int or there's another issue
                    competition = recent_match.get("competition", "")
                    response_parts.append(f"Their most recent match was against {opponent} in the {competition}")
        
        # Fix for the case where we have no valid response parts
        if not response_parts:
            return f"I have limited information about {team}, but they appear in our football database."
        
        return ". ".join(response_parts) + "."
    
    def _answer_general_league_question(self, question: str, leagues: List[str]) -> str:
        """Answer general questions about a league"""
        league = leagues[0]
        
        if league in self.league_tables:
            table = self.league_tables[league]
            
            if table:
                leader = table[0]
                leader_name = leader.get("team")
                leader_points = leader.get("points")
                
                relegation_zone = table[-3:]
                relegation_teams = [team.get("team") for team in relegation_zone]
                
                return f"In the {league}, {leader_name} is currently leading with {leader_points} points. The teams in the relegation zone are {', '.join(relegation_teams)}."
        
        return f"I don't have much information about the {league}."
    
    def _generate_fallback_response(self, question: str) -> str:
        """Generate a generic response when no specific answer can be found"""
        generic_responses = [
            "I don't have enough information to answer that question about football.",
            "I couldn't find the specific football information you're looking for.",
            "That's an interesting football question, but I don't have the data to answer it accurately.",
            "I'd need more specific details or different data to answer that football question.",
            "I don't have enough football data to provide a good answer to that question."
        ]
        
        # Simple hash of question to give consistent responses
        question_hash = sum(ord(c) for c in question) % len(generic_responses)
        return generic_responses[question_hash]
    
    def _answer_league_leaders(self, question: str, leagues: List[str]) -> str:
        """Answer questions about league leaders"""
        league = leagues[0]
        
        if league in self.league_tables and self.league_tables[league]:
            # Check if we're specifically asked for top 3 or similar
            question_lower = question.lower()
            if any(phrase in question_lower for phrase in ["top 3", "top three", "top 5", "top five"]):
                # Extract the number (3 or 5) from the question
                num_teams = 3  # Default
                if "top 5" in question_lower or "top five" in question_lower:
                    num_teams = 5
                
                top_teams = self.league_tables[league][:num_teams]
                
                team_descriptions = []
                for team in top_teams:
                    team_descriptions.append(f"{team.get('team')} ({team.get('position')}: {team.get('points')} pts)")
                
                return f"The top {num_teams} teams in the {league} are: {', '.join(team_descriptions)}."
            
            # Default case for league leader
            top_teams = self.league_tables[league][:3]  # Get top 3 teams
            
            if top_teams:
                leader = top_teams[0]
                leader_name = leader.get("team")
                leader_points = leader.get("points")
                leader_played = leader.get("played")
                
                response = f"{leader_name} is leading the {league} with {leader_points} points after {leader_played} games."
                
                if len(top_teams) > 1:
                    response += f" They are followed by {', '.join(t.get('team') for t in top_teams[1:])}."
                
                return response
        
        return f"I couldn't find information about the leaders in the {league}."

    def _handle_upcoming_matches(self, question: str, teams: List[str] = None, leagues: List[str] = None) -> str:
        # Filter upcoming matches
        filtered_matches = [m for m in self.upcoming_matches if m.get("is_upcoming", False)]
        
        if not filtered_matches:
            return "I don't have information about any upcoming matches."
        
        # Filter by team if specified
        if teams:
            team_matches = []
            for match in filtered_matches:
                home_team = match.get("home_team")
                away_team = match.get("away_team")
                if any(team == home_team or team == away_team for team in teams):
                    team_matches.append(match)
            
            if team_matches:
                if len(teams) == 1:
                    team = teams[0]
                    response = f"Upcoming matches for {team}:\n"
                    for match in team_matches:
                        home = match.get("home_team")
                        away = match.get("away_team")
                        competition = match.get("competition")
                        home_record = match.get("home_record", "N/A")
                        away_record = match.get("away_record", "N/A")
                        broadcast = ", ".join(match.get("broadcast", ["No broadcast information"]))
                        
                        response += f"• {home} vs {away} ({competition}) - {broadcast}\n"
                        response += f"  Records: {home} ({home_record}), {away} ({away_record})\n"
                    
                    return response.strip()
                else:
                    # Check for direct match between mentioned teams
                    for match in team_matches:
                        home = match.get("home_team")
                        away = match.get("away_team")
                        if home in teams and away in teams:
                            competition = match.get("competition")
                            home_record = match.get("home_record", "N/A")
                            away_record = match.get("away_record", "N/A")
                            broadcast = ", ".join(match.get("broadcast", ["No broadcast information"]))
                            
                            return f"Upcoming match: {home} vs {away} ({competition}) - {broadcast}\nRecords: {home} ({home_record}), {away} ({away_record})"
            
            return f"I couldn't find any upcoming matches for {'the teams' if len(teams) > 1 else teams[0]}."
        
        # Filter by league if specified
        if leagues:
            league = leagues[0]
            league_matches = [m for m in filtered_matches if m.get("competition") == league]
            
            if league_matches:
                response = f"Upcoming matches in {league}:\n"
                for match in league_matches:
                    home = match.get("home_team")
                    away = match.get("away_team")
                    broadcast = ", ".join(match.get("broadcast", ["No broadcast information"]))
                    
                    response += f"• {home} vs {away} - {broadcast}\n"
                
                return response.strip()
            
            return f"I couldn't find any upcoming matches for {league}."
        
        # Return all upcoming matches if no filter
        response = "Upcoming matches:\n"
        # Group by competition
        matches_by_competition = {}
        for match in filtered_matches[:10]:  # Limit to 10 matches for readability
            competition = match.get("competition")
            if competition not in matches_by_competition:
                matches_by_competition[competition] = []
            matches_by_competition[competition].append(match)
        
        for competition, matches in matches_by_competition.items():
            response += f"\n{competition}:\n"
            for match in matches:
                home = match.get("home_team")
                away = match.get("away_team")
                broadcast = ", ".join(match.get("broadcast", ["No broadcast information"]))
                
                response += f"• {home} vs {away} - {broadcast}\n"
        
        if len(filtered_matches) > 10:
            response += f"\n(Showing 10 of {len(filtered_matches)} upcoming matches)"
        
        return response.strip()

    def _handle_today_matches(self, question: str) -> str:
        """Handle questions about today's matches (both upcoming and completed)"""
        # For this function, we'll combine results from both matches and upcoming_matches
        # In a real system, you'd filter by date, but here we'll simulate that all matches in the data are for "today"
        
        response = ""
  
        upcoming_matches = [match for match in self.upcoming_matches if match['is_upcoming']]
        completed_matches = [match for match in self.upcoming_matches if not match['is_upcoming']]

        if not completed_matches and not upcoming_matches:
             response = "There are no matches played Today. "
        
        
        if completed_matches:
            response += "Here are today's completed matches:\n"
            # Group by competition
            matches_by_competition = {}
            for match in completed_matches:
                competition = match.get("competition")
                if competition not in matches_by_competition:
                    matches_by_competition[competition] = []
                matches_by_competition[competition].append(match)
            
            for competition, matches in matches_by_competition.items():
                response += f"\nIn the {competition}:\n"
                for match in matches:
                    home = match.get("home_team")
                    away = match.get("away_team")
                    home_score = match.get("home_score")
                    away_score = match.get("away_score")
                    
                    response += f"{home} {home_score} - {away_score} {away}\n"

        if upcoming_matches:
            response += "\nAs for the upcoming matches today:\n"
            # Group by competition
            matches_by_competition = {}
            for match in upcoming_matches:
                competition = match.get("competition")
                if competition not in matches_by_competition:
                    matches_by_competition[competition] = []
                matches_by_competition[competition].append(match)
            
            for competition, matches in matches_by_competition.items():
                response += f"\nIn the {competition}, we have:\n"
                for match in matches:
                    home = match.get("home_team")
                    away = match.get("away_team")
                    broadcast = ", ".join(match.get("broadcast", ["No broadcast information"]))
                    scheduled_time = match.get("scheduled_time", "Time TBD")
                    
                    response += f"{home} will face {away}\n"

        
        return response.strip()


