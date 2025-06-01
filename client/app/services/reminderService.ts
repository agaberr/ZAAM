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
  _id?: string; 
  title: string;
  description?: string;
  start_time: string;
  end_time?: string; 
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
  start_time: string;
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
      withCredentials: true, 
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
  console.log('[DEBUG] formatReminderForUI input:', apiReminder);
  
  try {
    // Parse the start_time and add 3 hours to match Egypt timezone
    const startTime = new Date(apiReminder.start_time);
    console.log('[DEBUG] Parsed start_time:', startTime);
    
    startTime.setHours(startTime.getHours() + 3);
    console.log('[DEBUG] Adjusted start_time (+3 hours):', startTime);
    
    const formatted = {
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
    
    console.log('[DEBUG] formatReminderForUI output:', formatted);
    return formatted;
  } catch (error) {
    console.error('[DEBUG] Error in formatReminderForUI:', error);
    console.error('[DEBUG] Input data was:', apiReminder);
    throw error;
  }
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
      
      // Backend returns {success, reminders, count, date} format or just {reminders}
      let reminders = [];
      
      if (response.data && response.data.success && response.data.reminders) {
        console.log(`[DEBUG] Found reminders in success format: ${response.data.reminders.length} items`);
        reminders = response.data.reminders;
      } else if (response.data && response.data.reminders && !response.data.success) {
        console.log(`[DEBUG] Found reminders without success flag: ${response.data.reminders.length} items`);
        reminders = response.data.reminders;
      } else if (Array.isArray(response.data)) {
        console.log(`[DEBUG] Response data is array: ${response.data.length} items`);
        reminders = response.data;
      } else {
        console.log('[DEBUG] No reminders found in getAllReminders, trying to find other formats...');
        console.log('[DEBUG] Response keys:', Object.keys(response.data || {}));
        reminders = [];
      }
      
      return reminders.map(formatReminderForUI);
    } catch (error) {
      console.error('Error fetching reminders:', error);
      throw error;
    }
  },
  
  // Get reminders for a specific date
  getRemindersForDate: async (date: string): Promise<ReminderType[]> => {
    try {
      console.log(`[DEBUG] Getting reminders for date: ${date}`);
      const api = await createAuthAPI();
      console.log('[DEBUG] API instance created, sending request...');
      
      // Calculate days offset from today
      const today = new Date();
      today.setHours(0, 0, 0, 0); // Reset to start of day
      
      const targetDate = new Date(date);
      targetDate.setHours(0, 0, 0, 0); // Reset to start of day
      
      const timeDiff = targetDate.getTime() - today.getTime();
      const daysDiff = Math.round(timeDiff / (1000 * 3600 * 24));
      
      console.log(`[DEBUG] Days offset: ${daysDiff} (from ${today.toDateString()} to ${targetDate.toDateString()})`);
      
      const response = await api.get(`/reminder?days_offset=${daysDiff}`);
      console.log('[DEBUG] Response received:', response.status);
      console.log('[DEBUG] Response data:', JSON.stringify(response.data, null, 2));
      
      // Backend returns {success, reminders, count, date} format or just {reminders}
      let reminders = [];
      
      if (response.data && response.data.success && response.data.reminders) {
        console.log(`[DEBUG] Found reminders in success format: ${response.data.reminders.length} items`);
        reminders = response.data.reminders;
      } else if (response.data && response.data.reminders && !response.data.success) {
        console.log(`[DEBUG] Found reminders without success flag: ${response.data.reminders.length} items`);
        reminders = response.data.reminders;
      } else if (Array.isArray(response.data)) {
        console.log(`[DEBUG] Response data is array: ${response.data.length} items`);
        reminders = response.data;
      } else {
        console.log('[DEBUG] No reminders found in response, trying to find other formats...');
        console.log('[DEBUG] Response keys:', Object.keys(response.data || {}));
        reminders = [];
      }
      
      console.log(`[DEBUG] Processing ${reminders.length} reminders`);
      
      const formattedReminders = reminders.map((reminder: any, index: number) => {
        console.log(`[DEBUG] Processing reminder ${index}:`, reminder);
        const formatted = formatReminderForUI(reminder);
        console.log(`[DEBUG] Formatted reminder ${index}:`, formatted);
        return formatted;
      });
      
      console.log(`[DEBUG] Final formatted reminders:`, formattedReminders);
      return formattedReminders;
      
    } catch (error: any) {
      const errorInfo = {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
        headers: error.response?.headers,
      };
      console.error('[DEBUG] Error fetching reminders for date:', JSON.stringify(errorInfo, null, 2));
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
  
  // Create a new reminder using AI endpoint with structured text
  createReminder: async (reminder: ReminderData): Promise<ReminderCreateResponse> => {
    try {
      const api = await createAuthAPI();
      
      // Build a structured text that the AI endpoint can process reliably
      let reminderText = `Remind me to ${reminder.title}`;
      
      if (reminder.start_time) {
        const startTime = new Date(reminder.start_time);
        
        // Format time in a clear format
        const timeStr = startTime.toLocaleTimeString('en-US', {
          hour: 'numeric',
          minute: '2-digit',
          hour12: true
        });
        
        // Check if it's today, tomorrow, or a specific date
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        
        const reminderDate = new Date(startTime);
        reminderDate.setHours(0, 0, 0, 0);
        
        const timeDiff = reminderDate.getTime() - today.getTime();
        const daysDiff = Math.round(timeDiff / (1000 * 3600 * 24));
        
        reminderText += ` at ${timeStr}`;
        
        if (daysDiff === 0) {
          reminderText += ` today`;
        } else if (daysDiff === 1) {
          reminderText += ` tomorrow`;
        } else if (daysDiff > 1) {
          const dateStr = startTime.toLocaleDateString('en-US', {
            weekday: 'long',
            month: 'long',
            day: 'numeric'
          });
          reminderText += ` on ${dateStr}`;
        }
      }

      console.log('[DEBUG] Creating reminder with AI text:', reminderText);
      
      const response = await api.post('/ai/reminder', { text: reminderText });
      
      console.log('[DEBUG] AI reminder response:', response.status, response.data);
      
      // Handle successful response
      if (response.data && response.data.success !== false) {
        return {
          message: response.data.response || response.data.message || 'Reminder created successfully',
          reminder_id: response.data.reminder?.id || response.data.reminder_id || 'unknown'
        };
      } else {
        throw new Error(response.data?.response || response.data?.message || 'Failed to create reminder');
      }
    } catch (error: any) {
      console.error('[DEBUG] Error creating reminder:', error);
      
      // Provide more detailed error information
      if (error.response) {
        console.error('[DEBUG] Response error details:', {
          status: error.response.status,
          data: error.response.data,
          headers: error.response.headers
        });
        
        // Extract error message from response
        const errorMessage = error.response.data?.response || 
                           error.response.data?.message || 
                           error.response.data?.error || 
                           `Server error: ${error.response.status}`;
        throw new Error(errorMessage);
      } else if (error.request) {
        throw new Error('Network error: Unable to reach server');
      } else {
        throw new Error(error.message || 'Unknown error occurred');
      }
    }
  },
  
  // Update an existing reminder
  updateReminder: async (reminderId: string, reminder: Partial<ReminderData>): Promise<void> => {
    try {
      console.log('[DEBUG] Updating reminder:', reminderId, reminder);
      const api = await createAuthAPI();
      
      // Format the data to match what the backend expects
      const updatePayload: any = {};
      
      if (reminder.title !== undefined) {
        updatePayload.title = reminder.title;
      }
      
      if (reminder.description !== undefined) {
        updatePayload.description = reminder.description;
      }
      
      if (reminder.start_time !== undefined) {
        // Convert ISO string to format Python can handle
        const startTime = new Date(reminder.start_time);
        updatePayload.start_time = startTime.toISOString().replace('Z', '+00:00');
      }
      
      if (reminder.end_time !== undefined) {
        // Convert ISO string to format Python can handle
        const endTime = new Date(reminder.end_time);
        updatePayload.end_time = endTime.toISOString().replace('Z', '+00:00');
      }
      
      if (reminder.completed !== undefined) {
        updatePayload.completed = reminder.completed;
      }
      
      // Note: recurrence is not handled by the backend PUT endpoint
      // If recurrence needs to be updated, we might need to use a different approach
      
      console.log('[DEBUG] Update payload:', updatePayload);
      
      const response = await api.put(`/reminder/${reminderId}`, updatePayload);
      
      console.log('[DEBUG] Update response:', response.status, response.data);
    } catch (error: any) {
      console.error('[DEBUG] Error updating reminder:', error);
      
      // Provide more detailed error information
      if (error.response) {
        console.error('[DEBUG] Update error details:', {
          status: error.response.status,
          data: error.response.data,
          headers: error.response.headers
        });
        
        // Extract error message from response
        const errorMessage = error.response.data?.error || 
                           error.response.data?.message || 
                           `Server error: ${error.response.status}`;
        throw new Error(errorMessage);
      } else if (error.request) {
        throw new Error('Network error: Unable to reach server');
      } else {
        throw new Error(error.message || 'Unknown error occurred');
      }
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
      console.log(`[DEBUG] Deleting reminder with ID: ${reminderId}`);
      const api = await createAuthAPI();
      
      // Validate reminder ID format
      if (!reminderId || reminderId.trim() === '') {
        throw new Error('Invalid reminder ID: empty or undefined');
      }
      
      console.log(`[DEBUG] Making DELETE request to: /reminder/${reminderId}`);
      const response = await api.delete(`/reminder/${reminderId}`);
      
      console.log(`[DEBUG] Delete reminder response:`, {
        status: response.status,
        data: response.data,
        headers: response.headers
      });
      
      // Check if deletion was successful
      if (response.status === 200 || response.status === 204) {
        console.log(`[DEBUG] Reminder deleted successfully`);
      } else {
        throw new Error(`Unexpected response status: ${response.status}`);
      }
      
    } catch (error: any) {
      console.error('[DEBUG] Error deleting reminder:', error);
      
      // Provide more detailed error information
      if (error.response) {
        console.error('[DEBUG] Delete error details:', {
          status: error.response.status,
          data: error.response.data,
          headers: error.response.headers,
          url: error.response.config?.url
        });
        
        // Extract error message from response
        const errorMessage = error.response.data?.error || 
                           error.response.data?.message || 
                           `Server error: ${error.response.status}`;
        throw new Error(errorMessage);
      } else if (error.request) {
        console.error('[DEBUG] Delete request error:', error.request);
        throw new Error('Network error: Unable to reach server');
      } else {
        console.error('[DEBUG] Delete setup error:', error.message);
        throw new Error(error.message || 'Unknown error occurred');
      }
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