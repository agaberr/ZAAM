import argparse
import requests
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime, timedelta

def authenticate(base_url, email, password):
    """Authenticate user and return token"""
    auth_url = f"{base_url}/api/auth/login"
    response = requests.post(auth_url, json={
        "email": email,
        "password": password
    })
    
    if response.status_code == 200:
        data = response.json()
        return data.get("token"), data.get("user_id")
    else:
        print(f"Authentication failed: {response.status_code}")
        print(response.text)
        return None, None

def get_detailed_stats(base_url, token):
    """Get detailed reminder statistics"""
    stats_url = f"{base_url}/api/reminders/stats/detailed"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(stats_url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get statistics: {response.status_code}")
        print(response.text)
        return None

def visualize_completion_rate(stats):
    """Visualize overall completion rate as a pie chart"""
    labels = ['Completed', 'Pending']
    sizes = [stats['completed_reminders'], stats['total_reminders'] - stats['completed_reminders']]
    colors = ['#4CAF50', '#FF9800']
    explode = (0.1, 0)
    
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title(f'Overall Completion Rate: {stats["completion_rate"]}%')
    plt.savefig('completion_rate.png')
    plt.close()

def visualize_reminders_by_type(stats):
    """Visualize reminders by type as a bar chart"""
    type_data = stats['reminders_by_type']
    categories = list(type_data.keys())
    values = list(type_data.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6'])
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom')
    
    plt.title('Reminders by Recurrence Type')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.savefig('reminders_by_type.png')
    plt.close()

def visualize_today_vs_upcoming(stats):
    """Visualize today's vs upcoming reminders"""
    categories = ['Today', 'Upcoming', 'Overdue']
    values = [stats['today_reminders'], stats['upcoming_reminders'], stats['overdue_reminders']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=colors)
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom')
    
    plt.title('Reminder Timeline')
    plt.xlabel('Time Period')
    plt.ylabel('Count')
    plt.savefig('reminder_timeline.png')
    plt.close()

def visualize_today_completion(stats):
    """Visualize today's completion rate"""
    labels = ['Completed Today', 'Pending Today']
    sizes = [stats['today_completed'], stats['today_reminders'] - stats['today_completed']]
    colors = ['#4CAF50', '#FF9800']
    explode = (0.1, 0)
    
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title(f"Today's Completion Rate: {stats['today_completion_rate']}%")
    plt.savefig('today_completion_rate.png')
    plt.close()

def visualize_recently_created(stats, days=30):
    """Visualize recently created reminders compared to all reminders"""
    labels = [f'Last {days} Days', 'Older']
    sizes = [stats['recently_created'], stats['total_reminders'] - stats['recently_created']]
    colors = ['#3498db', '#95a5a6']
    explode = (0.1, 0)
    
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Recently Created vs. Older Reminders')
    plt.savefig('recently_created.png')
    plt.close()

def generate_dashboard(stats):
    """Generate a dashboard with multiple visualizations"""
    # Create a 2x2 subplot
    plt.figure(figsize=(20, 15))
    
    # Overall completion rate
    plt.subplot(2, 2, 1)
    labels = ['Completed', 'Pending']
    sizes = [stats['completed_reminders'], stats['total_reminders'] - stats['completed_reminders']]
    colors = ['#4CAF50', '#FF9800']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title(f'Overall Completion Rate: {stats["completion_rate"]}%')
    
    # Reminders by type
    plt.subplot(2, 2, 2)
    type_data = stats['reminders_by_type']
    categories = list(type_data.keys())
    values = list(type_data.values())
    bars = plt.bar(categories, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom')
    plt.title('Reminders by Recurrence Type')
    
    # Today vs upcoming
    plt.subplot(2, 2, 3)
    categories = ['Today', 'Upcoming', 'Overdue']
    values = [stats['today_reminders'], stats['upcoming_reminders'], stats['overdue_reminders']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = plt.bar(categories, values, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom')
    plt.title('Reminder Timeline')
    
    # Today's completion rate
    plt.subplot(2, 2, 4)
    labels = ['Completed Today', 'Pending Today']
    sizes = [stats['today_completed'], max(0, stats['today_reminders'] - stats['today_completed'])]
    colors = ['#4CAF50', '#FF9800']
    if sum(sizes) > 0:  # Only create pie if there are reminders today
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
    else:
        plt.text(0.5, 0.5, "No reminders for today", ha='center', va='center')
    plt.axis('equal')
    plt.title(f"Today's Completion Rate: {stats['today_completion_rate']}%")
    
    # Add a title for the entire dashboard
    plt.suptitle(f"Reminder Statistics Dashboard - {datetime.now().strftime('%Y-%m-%d')}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('statistics_dashboard.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Reminder Statistics Visualization')
    parser.add_argument('--url', default='http://192.168.1.7:5000', help='Base URL of the API')
    parser.add_argument('--email', required=True, help='User email for authentication')
    parser.add_argument('--password', required=True, help='User password for authentication')
    parser.add_argument('--output', default='all', choices=['all', 'dashboard', 'completion', 'types', 'timeline', 'today', 'recent'],
                        help='Output visualization type')
    
    args = parser.parse_args()
    
    # Authenticate user
    token, user_id = authenticate(args.url, args.email, args.password)
    if not token:
        return
    
    # Get detailed statistics
    stats = get_detailed_stats(args.url, token)
    if not stats:
        return
    
    # Generate visualizations based on output type
    if args.output == 'all' or args.output == 'dashboard':
        generate_dashboard(stats)
        print("Generated dashboard visualization as statistics_dashboard.png")
    
    if args.output == 'all' or args.output == 'completion':
        visualize_completion_rate(stats)
        print("Generated completion rate visualization as completion_rate.png")
    
    if args.output == 'all' or args.output == 'types':
        visualize_reminders_by_type(stats)
        print("Generated reminders by type visualization as reminders_by_type.png")
    
    if args.output == 'all' or args.output == 'timeline':
        visualize_today_vs_upcoming(stats)
        print("Generated reminder timeline visualization as reminder_timeline.png")
    
    if args.output == 'all' or args.output == 'today':
        visualize_today_completion(stats)
        print("Generated today's completion rate visualization as today_completion_rate.png")
    
    if args.output == 'all' or args.output == 'recent':
        visualize_recently_created(stats)
        print("Generated recently created visualization as recently_created.png")

if __name__ == "__main__":
    main() 