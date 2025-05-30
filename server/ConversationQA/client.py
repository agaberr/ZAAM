# client.py
import requests
import json
import time

class ConversationalQAClient:
    def __init__(self, base_url="http://localhost:5003"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def ask_question(self, query):
        """Send a question to the QA system and get an answer"""
        url = f"{self.base_url}/api/query"
        data = {
            "query": query,
            "session_id": "user_session"  # Could be customized per user
        }
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying the QA system: {e}")
            return {"error": str(e), "status": "error"}
            
    def reset_conversation(self):
        """Reset the conversation history"""
        url = f"{self.base_url}/api/reset"
        
        try:
            response = self.session.post(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error resetting conversation: {e}")
            return {"error": str(e), "status": "error"}
            
    def get_history(self):
        """Get the conversation history"""
        url = f"{self.base_url}/api/history"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting history: {e}")
            return {"error": str(e), "status": "error"}
            
    def set_custom_passage(self, passage):
        """Set a custom passage for the QA system"""
        url = f"{self.base_url}/api/set_passage"
        data = {
            "passage": passage
        }
        
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error setting passage: {e}")
            return {"error": str(e), "status": "error"}


def interactive_demo():
    """Simple interactive demo of the QA client"""
    client = ConversationalQAClient()
    
    print("Conversational QA System Client")
    print("Type 'exit' to quit, 'reset' to reset conversation, 'history' to view history")
    
    while True:
        user_input = input("\nYour question: ")
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'reset':
            result = client.reset_conversation()
            print(f"Conversation reset: {result['status']}")
            continue
        elif user_input.lower() == 'history':
            history = client.get_history()
            print("\nConversation History:")
            for i, turn in enumerate(history['history']):
                print(f"{i+1}. Q: {turn['query']}")
                print(f"   A: {turn['answer']}")
                print(f"   Confidence: {turn.get('confidence', 'N/A')}")
            continue
        elif user_input.lower().startswith('passage:'):
            # Set custom passage
            passage = user_input[8:].strip()
            if passage:
                result = client.set_custom_passage(passage)
                print(f"Passage set: {result['status']}")
            else:
                print("No passage provided")
            continue
            
        # Send the question to the QA system
        start_time = time.time()
        result = client.ask_question(user_input)
        total_time = time.time() - start_time
        
        if result.get('status') == 'success':
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Client time: {total_time:.2f}s (Server processing: {result['processing_time']}s)")
            
            # Show debug info if available
            debug = result.get('debug', {})
            if debug.get('original_query') != debug.get('resolved_query'):
                print(f"Original question: {debug.get('original_query')}")
                print(f"Resolved question: {debug.get('resolved_query')}")
            if debug.get('current_entity'):
                print(f"Current entity: {debug.get('current_entity')}")
            if debug.get('needed_new_passage'):
                print("Note: A new passage was generated to answer this question")
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    interactive_demo()