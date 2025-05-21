import * as Google from 'expo-auth-session/providers/google';
import * as WebBrowser from 'expo-web-browser';
import { GoogleCalendar } from 'expo-calendar';
import api from './api';

WebBrowser.maybeCompleteAuthSession();

const CONFIG = {
  expoClientId: 'your-expo-client-id',
  androidClientId: 'your-android-client-id',
  iosClientId: 'your-ios-client-id',
  webClientId: 'your-web-client-id',
  scopes: ['https://www.googleapis.com/auth/calendar'],
};

export const useGoogleCalendar = () => {
  const [request, response, promptAsync] = Google.useAuthRequest({
    expoClientId: CONFIG.expoClientId,
    androidClientId: CONFIG.androidClientId,
    iosClientId: CONFIG.iosClientId,
    webClientId: CONFIG.webClientId,
    scopes: CONFIG.scopes,
  });

  const loginWithGoogle = async () => {
    try {
      // Get authorization URL from backend
      const { data } = await api.get('/api/auth/google/login');
      const { authorization_url } = data;
      
      // Open the authorization URL in browser
      const result = await WebBrowser.openAuthSessionAsync(
        authorization_url,
        'http://localhost:5000/callback'
      );
      
      if (result.type === 'success') {
        // Handle the callback URL
        const callbackUrl = result.url;
        await api.get(`/api/auth/google/callback?${callbackUrl.split('?')[1]}`);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error during Google login:', error);
      throw error;
    }
  };

  const checkGoogleAuth = async () => {
    try {
      const { data } = await api.get('/api/auth/google/check');
      return data.authenticated;
    } catch (error) {
      console.error('Error checking Google auth:', error);
      return false;
    }
  };

  const logoutGoogle = async () => {
    try {
      await api.get('/api/auth/google/logout');
      return true;
    } catch (error) {
      console.error('Error during Google logout:', error);
      throw error;
    }
  };

  const addEventToCalendar = async (eventDetails: {
    title: string;
    description?: string;
    startDate: Date;
    endDate: Date;
    location?: string;
  }) => {
    try {
      const calendar = await GoogleCalendar.createEventAsync(
        'primary',
        {
          title: eventDetails.title,
          notes: eventDetails.description,
          startDate: eventDetails.startDate,
          endDate: eventDetails.endDate,
          location: eventDetails.location,
          timeZone: 'UTC',
        }
      );
      
      return calendar;
    } catch (error) {
      console.error('Error adding event to calendar:', error);
      throw error;
    }
  };

  const getCalendarEvents = async (startDate: Date, endDate: Date) => {
    try {
      const events = await GoogleCalendar.getEventsAsync(
        'primary',
        startDate,
        endDate
      );
      
      return events;
    } catch (error) {
      console.error('Error fetching calendar events:', error);
      throw error;
    }
  };

  return {
    request,
    response,
    promptAsync,
    loginWithGoogle,
    checkGoogleAuth,
    logoutGoogle,
    addEventToCalendar,
    getCalendarEvents,
  };
};
