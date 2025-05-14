import axios from 'axios';
import * as SecureStore from 'expo-secure-store';

// const API_URL = 'https://zaam-mj7u.onrender.com/api';
const API_URL = 'http://34.57.245.214:5000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add authentication token to requests
api.interceptors.request.use(
  async (config) => {
    const token = await SecureStore.getItemAsync('userToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Authentication service
export const authService = {
  login: async (email: string, password: string) => {
    const response = await api.post('/auth/login', { email, password });
    return response.data;
  },
  
  register: async (userData: any) => {
    const response = await api.post('/auth/register', userData);
    return response.data;
  },
  
  logout: async () => {
    await SecureStore.deleteItemAsync('userToken');
  },
};

// User service
export const userService = {
  getProfile: async () => {
    const response = await api.get('/users/profile');
    return response.data;
  },
  
  updateProfile: async (updates: any) => {
    const response = await api.put('/users/profile', updates);
    return response.data;
  },
};

// Reminders service
export const reminderService = {
  getReminders: async () => {
    const response = await api.get('/reminders');
    return response.data;
  },
  
  addReminder: async (reminderData: any) => {
    const response = await api.post('/reminders', reminderData);
    return response.data;
  },
  
  updateReminderStatus: async (reminderId: string, status: string) => {
    const response = await api.patch(`/reminders/${reminderId}`, { status });
    return response.data;
  },
  
  deleteReminder: async (reminderId: string) => {
    const response = await api.delete(`/reminders/${reminderId}`);
    return response.data;
  },
};

// Chat service
export const chatService = {
  getChatHistory: async () => {
    const response = await api.get('/chats/history');
    return response.data;
  },
  
  sendMessage: async (message: string) => {
    const response = await api.post('/chats/message', { message });
    return response.data;
  },
};

export default api;
