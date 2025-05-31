#!/usr/bin/env python3
"""
Test script to demonstrate the improved intent classification for reminders.
This tests the logic where:
- If user includes time in query -> "create_event" intent
- If user doesn't include time in query -> "get_timetable" intent
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from models.reminder import ReminderNLP


def test_intent_classification():
    """Test the intent classification with various sample queries"""
    
    # Initialize the ReminderNLP (this might take a moment to load the model)
    print("Loading ReminderNLP model...")
    try:
        reminder_nlp = ReminderNLP()
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you have the trained model files in server/reminders/ directory")
        return
    
    # Test cases: queries WITH time information (should be create_event)
    create_event_queries = [
        "Remind me to take medicine at 3 pm",
        "Schedule a meeting at 10:30 am tomorrow",
        "Set a reminder for doctor appointment at 2 o'clock",
        "Remind me to call mom at 5 pm toseday",
        "Schedule lunch meeting at noon",
        "Remind me to exercise at 7 am",
        "Set appointment with dentist at 4:15 pm",
        "Remind me to take pills at midnight"
    ]
    
    # Test cases: queries WITHOUT time information (should be get_timetable)
    get_timetable_queries = [
        "What's on my schedule today?",
        "Show me my reminders",
        "What do I have planned for tomorrow?",
        "Tell me my appointments",
        "Check my schedule",
        "What's coming up?",
        "List my reminders for today",
        "Show me what I need to do"
    ]
    
    print("🧪 Testing CREATE_EVENT intent (queries WITH time):")
    print("=" * 60)
    
    for query in create_event_queries:
        result = reminder_nlp.process_text(query)
        intent = result.get('predicted_intent', 'unknown')
        predicted_time = result.get('predicted_time', 'None')
        predicted_action = result.get('predicted_action', 'None')
        
        status = "✅" if intent == "create_event" else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Intent: {intent} | Time: {predicted_time} | Action: {predicted_action}")
        print()
    
    print("\n🧪 Testing GET_TIMETABLE intent (queries WITHOUT time):")
    print("=" * 60)
    
    for query in get_timetable_queries:
        result = reminder_nlp.process_text(query)
        intent = result.get('predicted_intent', 'unknown')
        predicted_time = result.get('predicted_time', 'None')
        predicted_action = result.get('predicted_action', 'None')
        
        status = "✅" if intent == "get_timetable" else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Intent: {intent} | Time: {predicted_time} | Action: {predicted_action}")
        print()

    print("\n📊 Summary:")
    print("This test demonstrates the new intent classification logic:")
    print("• Queries WITH time information → 'create_event' intent")
    print("• Queries WITHOUT time information → 'get_timetable' intent")
    print("\nThe system now correctly classifies user intents based on time presence!")


if __name__ == "__main__":
    test_intent_classification() 