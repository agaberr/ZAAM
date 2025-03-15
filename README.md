# Alzheimer's AI Companion App

## Overview
The **Alzheimer's AI Companion App** is designed to assist Alzheimer's patients by providing an AI-powered interactive companion that helps with daily tasks, reminders, and cognitive engagement. Additionally, it includes a **dashboard for caregivers** to monitor patient activities and AI interactions.

## Tech Stack
- **Frontend**: React Native with TypeScript, Expo, and Expo Router
- **Backend**: Node.js
- **UI Framework**: React Native Paper
- **Database**: MongoDB 

## App Flow

### 1. Welcome Screen
- A clean and minimalistic interface featuring the app logo.
- Options to **Sign Up** or **Log In**.
- Users can sign up via **Email** or **Google** authentication.

### 2. User Data Creation
After signing up, users are prompted to enter essential information:
- **Name**
- **Gender**
- **Email**
- **Medications Used**
- **Additional Medical Information** (if applicable)
- **Caregiver's Contact Details** (optional)
- Submission redirects users to the **Main Dashboard**.

### 3. Main Dashboard
The central hub displaying patient activities and AI interaction data, including:
- **Activity Logs**: History of patient interactions with AI.
- **AI Conversation Analysis**: Summarized insights from AI interactions.
- **Daily Routine Summary**: Overview of scheduled tasks and reminders.
- **Health & Medication Tracking**: Displays upcoming and missed medications.

### 4. Reminders Page (Google Calendar Integration)
- **View all reminders** from Google Calendar.
- **Add new reminders** (medications, appointments, daily activities).
- **Edit/Delete reminders**.
- Automatic **synchronization** with Google Calendar.

### 5. Talk to AI (3D Rendered AI)
A dedicated interactive page where users can engage with their AI companion:
- The AI model is **3D-rendered** and can be cast to a **TV** for a more immersive experience.
- The AI can **respond, provide reminders, engage in conversations**, and assist with everyday tasks.
- Features include:
  - **Speech-to-Text & Text-to-Speech Processing**
  - **Natural Language Processing (NLP)** for context-aware conversations
  - **Task execution** (e.g., calling someone, fetching news, setting reminders)

### 6. Profile & Settings Page
- View and **update personal data** (Name, Gender, Medications, Contact details, etc.).
- Manage **AI interaction preferences**.
- **Account settings** (password change, logout, notification preferences).

---

## Database Schema

### User Schema
```json
{
    "_id": "ObjectId",
    "full_name": "John Doe",
    "age": 75,
    "gender": "Male",
    "contact_info": {
        "phone": "+1234567890",
        "email": "johndoe@example.com"
    },
    "emergency_contacts": [
        {
            "name": "Jane Doe",
            "relationship": "Daughter",
            "phone": "+9876543210"
        }
    ],
    "preferences": {
        "language": "English",
        "voice_type": "Male",
        "reminder_frequency": "hourly"
    },
    "created_at": "ISODate",
    "updated_at": "ISODate"
}
```

### Chat Logs Schema
```json
{
    "_id": "ObjectId",
    "user_id": "ObjectId",
    "conversation": [
        {
            "timestamp": "ISODate",
            "sender": "user",
            "message": "Hello, how are you?",
            "emotion_detected": "neutral"
        },
        {
            "timestamp": "ISODate",
            "sender": "AI",
            "message": "I am here to assist you. How can I help?",
            "response_type": "informational"
        }
    ],
    "summary": "User asked how AI is doing. AI responded positively.",
    "created_at": "ISODate"
}
```

### Schedules Schema
```json
{
    "_id": "ObjectId",
    "user_id": "ObjectId",
    "reminders": [
        {
            "type": "medication",
            "name": "Aspirin",
            "dosage": "100mg",
            "time": "08:00 AM",
            "recurrence": "daily",
            "status": "pending",
            "last_taken": "ISODate"
        },
        {
            "type": "hydration",
            "time": "10:00 AM",
            "recurrence": "every 2 hours",
            "status": "completed"
        }
    ],
    "created_at": "ISODate",
    "updated_at": "ISODate"
}
```

---

## Optimal Folder Structure

## Technical Components

### 1. Authentication
- Email & Password authentication.
- Google OAuth for quick sign-in.

### 2. Database & Storage
- **NoSQL Database** for storing user data, schedules, and chat history.
- **Vector Database** for improving NLP accuracy in AI responses.

### 3. AI & NLP Processing
- Speech-to-text processing for verbal interactions.
- AI **categorizes and queues** tasks for execution.
- Utilizes **contextual awareness** to provide meaningful responses.

### 4. Google Calendar API Integration
- Fetch, add, edit, and delete reminders seamlessly.

### 5. 3D AI Rendering & TV Casting
- The AI avatar is **rendered in 3D** for a lifelike experience.
- Supports **TV casting** for a larger display.

### 6. Dashboard for Caregivers
- Provides a caregiver-friendly view of patient activity and AI interactions.
- Enables caregivers to **monitor patient well-being** remotely.

---

## Conclusion
This app aims to **enhance the lives of Alzheimer's patients** by providing an intelligent AI companion while giving caregivers insights into patient activities. The integration of **NLP, Google Calendar, and 3D AI rendering** makes this a comprehensive solution for daily assistance and cognitive support.