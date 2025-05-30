import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { format } from 'date-fns';
import { formatInTimeZone } from 'date-fns-tz';

// Base API URL (should match your server configuration)
// const API_URL = 'https://zaam-mj7u.onrender.com/api';
const API_URL = 'http://localhost:5003/api';

const EGYPT_TZ = 'Africa/Cairo';

// Statistics interface
export interface ReminderStats {
  totalCount: number;
  completedCount: number;
  pendingCount: number;
  upcomingCount: number;
  completionRate: number;
  byType: {
    medication: number;
    appointment: number;
    activity: number;
    hydration: number;
  };
}

// Reminder interface that matches the server model
export interface ReminderData {
  _id?: string; // MongoDB ID (undefined for new reminders)
  title: string;
  description?: string;
  start_time: string; // ISO date string
  end_time?: string; // ISO date string or null
  recurrence?: 'daily' | 'weekly' | 'monthly' | 'yearly' | null;
  completed?: boolean;
}

export interface ReminderCreateResponse {
  message: string;
  reminder_id: string;
}

export interface ReminderType {
  id: string;
  title: string;
  time: string;
  date: string;
  start_time: string; // Keep original ISO string for editing
  type: 'medication' | 'appointment' | 'activity' | 'hydration';
  description?: string;
  location?: string;
  completed: boolean;
  recurring?: boolean;
  recurrencePattern?: string;
}

// Get auth token from async storage
const getAuthToken = async (): Promise<string | null> => {
  try {
    return await AsyncStorage.getItem('authToken');
  } catch (error) {
    console.error('Error getting auth token:', error);
    return null;
  }
};

// Create axios instance with authorization headers
const createAuthAPI = async () => {
  try {
    const token = await getAuthToken();
    
    return axios.create({
      baseURL: API_URL,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': token ? `Bearer ${token}` : '',
      },
      withCredentials: true, // Include cookies in requests
    });
  } catch (error) {
    console.error('Error creating auth API:', error);
    // Return a default instance without auth headers as fallback
    return axios.create({
      baseURL: API_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }
};

// Helper to convert API reminders to UI format
const formatReminderForUI = (apiReminder: any): ReminderType => {
  // Parse the start_time and add 3 hours to match Egypt timezone
  const startTime = new Date(apiReminder.start_time);
  startTime.setHours(startTime.getHours() + 3);
  
  return {
    id: apiReminder._id || '',
    title: apiReminder.title,
    time: formatInTimeZone(startTime.toISOString(), EGYPT_TZ, 'hh:mm a'),
    date: formatInTimeZone(startTime.toISOString(), EGYPT_TZ, 'yyyy-MM-dd'),
    start_time: apiReminder.start_time, // Keep original ISO string for editing
    type: mapReminderToType(apiReminder),
    description: apiReminder.description,
    completed: apiReminder.completed ?? false, // Handle undefined completed field
    recurring: !!apiReminder.recurrence,
    recurrencePattern: formatRecurrencePattern(apiReminder.recurrence),
  };
};

// Map reminder to a type based on its content
const mapReminderToType = (reminder: any): 'medication' | 'appointment' | 'activity' | 'hydration' => {
  const title = reminder.title.toLowerCase();
  
  // Basic classification logic - can be improved with better AI/matching
  if (title.includes('medicine') || title.includes('pill') || title.includes('medication') || title.includes('take')) {
    return 'medication';
  }
  
  if (title.includes('doctor') || title.includes('appointment') || title.includes('visit') || title.includes('check-up')) {
    return 'appointment';
  }
  
  if (title.includes('water') || title.includes('drink') || title.includes('hydrate')) {
    return 'hydration';
  }
  
  return 'activity'; // Default type
};

// Format recurrence pattern for display
const formatRecurrencePattern = (recurrence: string | null | undefined): string => {
  if (!recurrence) return '';
  
  switch (recurrence) {
    case 'daily': return 'Daily';
    case 'weekly': return 'Weekly';
    case 'monthly': return 'Monthly';
    case 'yearly': return 'Yearly';
    default: return recurrence;
  }
};

// API functions
export const reminderService = {
  // Get all reminders for the authenticated user
  getAllReminders: async (): Promise<ReminderType[]> => {
    try {
      const api = await createAuthAPI();
      const response = await api.get('/reminder');
      
      // Backend returns {success, reminders, count, date} format
      if (response.data && response.data.success && response.data.reminders) {
        return response.data.reminders.map(formatReminderForUI);
      } else if (Array.isArray(response.data)) {
        // Fallback if backend returns array directly
        return response.data.map(formatReminderForUI);
      } else {
        console.error('Unexpected response format:', response.data);
        return [];
      }
    } catch (error) {
      console.error('Error fetching reminders:', error);
      throw error;
    }
  },
  
  // Get reminders for a specific date
  getRemindersForDate: async (date: string): Promise<ReminderType[]> => {
    try {
      console.log(`Getting reminders for date: ${date}`);
      const api = await createAuthAPI();
      console.log('API instance created, sending request...');
      
      // Calculate days offset from today
      const today = new Date();
      today.setHours(0, 0, 0, 0); // Reset to start of day
      
      const targetDate = new Date(date);
      targetDate.setHours(0, 0, 0, 0); // Reset to start of day
      
      const timeDiff = targetDate.getTime() - today.getTime();
      const daysDiff = Math.round(timeDiff / (1000 * 3600 * 24));
      
      console.log(`Days offset: ${daysDiff} (from ${today.toDateString()} to ${targetDate.toDateString()})`);
      
      const response = await api.get(`/reminder?days_offset=${daysDiff}`);
      console.log('Response received:', response.status);
      
      // Backend returns {success, reminders, count, date} format
      if (response.data && response.data.success && response.data.reminders) {
        console.log(`Received ${response.data.reminders.length} reminders`);
        return response.data.reminders.map(formatReminderForUI);
      } else if (Array.isArray(response.data)) {
        // Fallback if backend returns array directly
        console.log(`Received ${response.data.length} reminders`);
        return response.data.map(formatReminderForUI);
      } else {
        console.error('Invalid response format:', response.data);
        return [];
      }
    } catch (error: any) {
      const errorInfo = {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
        headers: error.response?.headers,
      };
      console.error('Error fetching reminders for date:', JSON.stringify(errorInfo, null, 2));
      throw error;
    }
  },
  
  // Get today's reminders
  getTodayReminders: async (): Promise<ReminderType[]> => {
    try {
      const api = await createAuthAPI();
      // Use days_offset=0 for today
      const response = await api.get('/reminder?days_offset=0');
      
      // Backend returns {success, reminders, count, date} format
      if (response.data && response.data.success && response.data.reminders) {
        return response.data.reminders.map(formatReminderForUI);
      } else if (Array.isArray(response.data)) {
        // Fallback if backend returns array directly
        return response.data.map(formatReminderForUI);
      } else {
        console.error('Unexpected response format:', response.data);
        return [];
      }
    } catch (error) {
      console.error('Error fetching today\'s reminders:', error);
      throw error;
    }
  },
  
  // Create a new reminder using AI endpoint
  createReminder: async (reminder: ReminderData): Promise<ReminderCreateResponse> => {
    try {
      const api = await createAuthAPI();
      
      // Parse the start_time and subtract 3 hours to compensate for the timezone difference
      // This ensures that when the backend adds 3 hours, it will be the correct time
      const startTime = new Date(reminder.start_time);
      startTime.setHours(startTime.getHours() - 3);
      
      // Format time string in 12-hour format
      const timeStr = startTime.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      });
      
      const dateStr = startTime.toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
      
      let reminderText = `Remind me to ${reminder.title}`;
      if (reminder.start_time) {
        reminderText += ` at ${timeStr}`;
        // Check if it's not today
        const today = new Date();
        const reminderDate = startTime;
        if (reminderDate.toDateString() !== today.toDateString()) {
          reminderText += ` on ${dateStr}`;
        }
      }
      
      const response = await api.post('/ai/reminder', { text: reminderText });
      
      if (response.data && response.data.success) {
        return {
          message: response.data.message || 'Reminder created successfully',
          reminder_id: response.data.reminder?.id || 'unknown'
        };
      } else {
        throw new Error(response.data?.response || 'Failed to create reminder');
      }
    } catch (error) {
      console.error('Error creating reminder:', error);
      throw error;
    }
  },
  
  // Update an existing reminder
  updateReminder: async (reminderId: string, reminder: Partial<ReminderData>): Promise<void> => {
    try {
      const api = await createAuthAPI();
      await api.put(`/reminder/${reminderId}`, reminder);
    } catch (error) {
      console.error('Error updating reminder:', error);
      throw error;
    }
  },
  
  // Toggle completion status
  toggleCompletion: async (reminderId: string, completed: boolean): Promise<void> => {
    try {
      console.log(`[DEBUG] Toggling reminder ${reminderId} to completed: ${completed}`);
      const api = await createAuthAPI();
      const response = await api.put(`/reminder/${reminderId}`, { completed });
      console.log(`[DEBUG] Toggle completion response:`, response.status, response.data);
    } catch (error) {
      console.error('Error toggling reminder completion:', error);
      throw error;
    }
  },
  
  // Delete a reminder
  deleteReminder: async (reminderId: string): Promise<void> => {
    try {
      console.log(`[DEBUG] Deleting reminder ${reminderId}`);
      const api = await createAuthAPI();
      const response = await api.delete(`/reminder/${reminderId}`);
      console.log(`[DEBUG] Delete reminder response:`, response.status, response.data);
    } catch (error) {
      console.error('Error deleting reminder:', error);
      throw error;
    }
  },
  
  // Set voice/AI reminder
  setVoiceReminder: async (text: string): Promise<{success: boolean, message: string}> => {
    try {
      const api = await createAuthAPI();
      const response = await api.post('/ai/reminder', { text });
      
      return {
        success: response.data.success || false,
        message: response.data.message || response.data.response || 'Reminder created'
      };
    } catch (error) {
      console.error('Error setting voice reminder:', error);
      throw error;
    }
  },
  
  // Get global reminder statistics (calculated from all reminders since backend doesn't have stats endpoint)
  getStatistics: async (): Promise<ReminderStats> => {
    try {
      // Since backend doesn't have /reminder/stats endpoint, calculate from all reminders
      const allReminders = await reminderService.getAllReminders();
      
      const completedCount = allReminders.filter(r => r.completed).length;
      const pendingCount = allReminders.filter(r => !r.completed).length;
      
      // Calculate counts by type
      const byType = {
        medication: allReminders.filter(r => r.type === 'medication').length,
        appointment: allReminders.filter(r => r.type === 'appointment').length,
        activity: allReminders.filter(r => r.type === 'activity').length,
        hydration: allReminders.filter(r => r.type === 'hydration').length,
      };
      
      return {
        totalCount: allReminders.length,
        completedCount,
        pendingCount,
        upcomingCount: pendingCount, // Same as pending for now
        completionRate: allReminders.length ? Math.round((completedCount / allReminders.length) * 100) : 0,
        byType
      };
    } catch (error) {
      console.error('Error fetching reminder statistics:', error);
      
      // Return default stats on error
      return {
        totalCount: 0,
        completedCount: 0,
        pendingCount: 0,
        upcomingCount: 0,
        completionRate: 0,
        byType: {
          medication: 0,
          appointment: 0,
          activity: 0,
          hydration: 0
        }
      };
    }
  }
}; 