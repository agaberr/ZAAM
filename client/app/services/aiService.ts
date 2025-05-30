import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Base API URL (should match your server configuration)
const API_URL = 'http://localhost:5003/api';

// AI Response interface
export interface AIResponse {
  response: string;
  category_responses?: Record<string, string>;
  categories?: string[];
  success: boolean;
  audio?: string; // Add audio field for base64 audio data
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

export const aiService = {
  // Process AI request with proper authentication
  processAIRequest: async (text: string): Promise<AIResponse> => {
    try {
      const api = await createAuthAPI();
      const response = await api.post('/ai/process', { text: text.trim() });
      
      if (response.data) {
        return {
          response: response.data.response || "I'm here to help you. What would you like to know?",
          category_responses: response.data.category_responses,
          categories: response.data.categories,
          success: response.data.success || false,
          audio: response.data.audio_data // Include audio data from the server
        };
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Error processing AI request:', error);
      throw error;
    }
  },

  // Process news-specific requests
  processNewsRequest: async (text: string): Promise<AIResponse> => {
    try {
      const api = await createAuthAPI();
      const response = await api.post('/ai/news', { text: text.trim() });
      
      return {
        response: response.data.response || "Sorry, I couldn't process your news request.",
        success: response.data.success || false
      };
    } catch (error) {
      console.error('Error processing news request:', error);
      throw error;
    }
  },

  // Process weather-specific requests
  processWeatherRequest: async (text: string): Promise<AIResponse> => {
    try {
      const api = await createAuthAPI();
      const response = await api.post('/ai/weather', { text: text.trim() });
      
      return {
        response: response.data.response || "Sorry, I couldn't get weather information.",
        success: response.data.success || false
      };
    } catch (error) {
      console.error('Error processing weather request:', error);
      throw error;
    }
  },

  // Process reminder-specific requests
  processReminderRequest: async (text: string): Promise<AIResponse> => {
    try {
      const api = await createAuthAPI();
      const response = await api.post('/ai/reminder', { text: text.trim() });
      
      return {
        response: response.data.response || "Sorry, I couldn't process your reminder request.",
        success: response.data.success || false
      };
    } catch (error) {
      console.error('Error processing reminder request:', error);
      throw error;
    }
  }
}; 