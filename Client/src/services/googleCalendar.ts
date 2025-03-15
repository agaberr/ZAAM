import * as Google from 'expo-auth-session/providers/google';
import * as WebBrowser from 'expo-web-browser';
import { GoogleCalendar } from 'expo-calendar';

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
    addEventToCalendar,
    getCalendarEvents,
  };
};
