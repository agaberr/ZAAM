import React, { createContext, useState, useContext, useEffect } from 'react';
import { router, useSegments, useRootNavigationState } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as WebBrowser from 'expo-web-browser';
import * as Google from 'expo-auth-session/providers/google';
import axios from 'axios';

// API endpoint base URL
const API_BASE_URL = 'http://localhost:5000'; // For Android emulator pointing to localhost
// const API_BASE_URL = 'http://10.0.2.2:5000'; // For Android emulator pointing to localhost

// If using a physical device, use your computer's IP address instead, e.g. 'http://192.168.1.100:5000'

// Define the shape of the auth context
type AuthContextType = {
  isAuthenticated: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  completeOnboarding: (userData: UserDataType) => Promise<void>;
  fetchUserData: (userId: string) => Promise<any>;
  updateUserProfile: (userId: string, userData: any) => Promise<void>;
  fetchMemoryAids: () => Promise<any[]>;
  createMemoryAid: (memoryAidData: MemoryAidDataType) => Promise<any>;
  updateMemoryAid: (memoryAidId: string, memoryAidData: Partial<MemoryAidDataType>) => Promise<any>;
  deleteMemoryAid: (memoryAidId: string) => Promise<boolean>;
  userData: any | null;
  tempRegData: TempRegDataType | null;
  googleCredentials: any | null;
  googleSignIn: () => Promise<void>;
  googleSignOut: () => Promise<void>;
  createCalendarEvent: (title: string, description: string, startTime: Date) => Promise<any>;
  getCalendarEvents: (startDate?: Date, endDate?: Date) => Promise<any[]>;
  processNaturalLanguageReminder: (text: string) => Promise<any>;
};

// Define temp registration data type
type TempRegDataType = {
  full_name: string;
  email: string;
  password: string;
};

// Define user data type for onboarding
type EmergencyContactType = {
  name: string;
  relationship: string;
  phone: string;
};

type UserDataType = {
  age: number;
  gender: string;
  phone: string;
  emergency_contacts: EmergencyContactType[];
};

// Define memory aid data type
type MemoryAidDataType = {
  title: string;
  description: string;
  type: 'person' | 'place' | 'event' | 'object';
  date?: string;
  image_url?: string;
};

// Create the context with a default value
const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  signIn: async () => {},
  signUp: async () => {},
  signOut: async () => {},
  completeOnboarding: async () => {},
  fetchUserData: async () => {},
  updateUserProfile: async () => {},
  fetchMemoryAids: async () => [],
  createMemoryAid: async () => ({}),
  updateMemoryAid: async () => ({}),
  deleteMemoryAid: async () => false,
  userData: null,
  tempRegData: null,
  googleCredentials: null,
  googleSignIn: async () => {},
  googleSignOut: async () => {},
  createCalendarEvent: async () => ({}),
  getCalendarEvents: async () => [],
  processNaturalLanguageReminder: async () => ({}),
});

// Custom hook to use the auth context
export const useAuth = () => useContext(AuthContext);

// Provider component that wraps your app and makes auth object available to any child component
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [userData, setUserData] = useState<any | null>(null);
  const [tempRegData, setTempRegData] = useState<TempRegDataType | null>(null);
  const [googleCredentials, setGoogleCredentials] = useState<any | null>(null);
  
  const segments = useSegments();
  const navigationState = useRootNavigationState();

  // Google authentication setup
  const [request, response, promptAsync] = Google.useAuthRequest({
    clientId: '485643114726-0joadin731ltorui0db3v6dma86rjbvb.apps.googleusercontent.com',
    scopes: ['profile', 'email', 'https://www.googleapis.com/auth/calendar'],
    redirectUri: 'exp://localhost:19000/--/callback'
  });

  // Check if the user is authenticated on initial load
  useEffect(() => {
    const loadAuthState = async () => {
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        const userDataString = await AsyncStorage.getItem('userData');
        const tempRegDataString = await AsyncStorage.getItem('tempRegData');
        const googleCredentialsString = await AsyncStorage.getItem('googleCredentials');
        
        if (authToken && userDataString) {
          const parsedUserData = JSON.parse(userDataString);
          setUserData(parsedUserData);
          setIsAuthenticated(true);
        }
        
        if (tempRegDataString) {
          setTempRegData(JSON.parse(tempRegDataString));
        }
        
        if (googleCredentialsString) {
          setGoogleCredentials(JSON.parse(googleCredentialsString));
        }
      } catch (error) {
        console.error('Failed to load auth state:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadAuthState();
  }, []);

  // Handle routing based on authentication state
  useEffect(() => {
    // Don't run navigation if the app is still loading or navigation is not ready
    if (isLoading || !navigationState?.key) return;
    
    const inAuthGroup = segments[0] === 'auth';
    const inOnboardingGroup = segments[0] === 'onboarding';
    const isWelcomeScreen = segments[0] === 'welcome';
    const isIndexScreen = segments[0] === undefined || segments[0] === '';
    
    // Use a setTimeout to ensure navigation happens after the layout is fully mounted
    setTimeout(() => {
      if (isIndexScreen && !isAuthenticated) {
        // Redirect to welcome screen if on index and not authenticated
        router.replace('/welcome');
      } else if (!isAuthenticated && !inAuthGroup && !inOnboardingGroup && !isWelcomeScreen) {
        // Redirect to the welcome screen if not authenticated and not already on welcome/auth/onboarding
        router.replace('/welcome');
      } else if (isAuthenticated && (inAuthGroup || isWelcomeScreen)) {
        // Redirect to the home screen if authenticated and on auth/welcome screens
        router.replace('/');
      }
    }, 0);
  }, [isAuthenticated, segments, navigationState?.key, isLoading]);

  // Handle Google auth response
  useEffect(() => {
    if (response?.type === 'success') {
      const { authentication } = response;
      setGoogleCredentials(authentication);
      saveGoogleCredentials(authentication);
    }
  }, [response]);

  // Save Google credentials to storage
  const saveGoogleCredentials = async (credentials: any) => {
    try {
      await AsyncStorage.setItem('googleCredentials', JSON.stringify(credentials));
    } catch (error) {
      console.error('Failed to save Google credentials:', error);
    }
  };

  // Google sign in
  const googleSignIn = async () => {
    try {
      setIsLoading(true);
      await promptAsync();
    } catch (error) {
      console.error('Google sign in error:', error);
      setIsAuthenticated(false);
      setUserData(null);
      setGoogleCredentials(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Google sign out
  const googleSignOut = async () => {
    try {
      setIsLoading(true);
      await AsyncStorage.removeItem('googleCredentials');
      setGoogleCredentials(null);
      setIsAuthenticated(false);
      setUserData(null);
    } catch (error) {
      console.error('Google sign out error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Process natural language reminder request
  const processNaturalLanguageReminder = async (text: string) => {
    try {
      if (!text.trim()) {
        throw new Error("Text cannot be empty");
      }
      
      const authToken = await AsyncStorage.getItem('authToken');
      if (!authToken) {
        throw new Error("Authentication token not found");
      }
      
      // Send text to backend NLP processor
      const response = await axios.post(`${API_BASE_URL}/api/reminders/process-reminder`, 
        { text },
        {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        }
      );
      
      const { intent, slots } = response.data;
      
      // If intent is to create an event, handle it
      if (intent === 'create_event' && slots.action) {
        // Create event in calendar
        if (googleCredentials) {
          // Parse time from slots or default to now
          let startTime = new Date();
          if (slots.time) {
            // Simple parsing for common time formats
            const timeStr = slots.time.toLowerCase();
            if (timeStr.includes('am') || timeStr.includes('pm')) {
              const [hourStr, period] = timeStr.split(/\s+/);
              let hour = parseInt(hourStr);
              if (period === 'pm' && hour < 12) hour += 12;
              if (period === 'am' && hour === 12) hour = 0;
              startTime.setHours(hour, 0, 0, 0);
            }
          }
          
          // Create calendar event
          return await createCalendarEvent(
            slots.action,
            `Created via voice command: "${text}"`,
            startTime
          );
        } else {
          throw new Error("Google Calendar authentication required");
        }
      }
      
      return { intent, slots };
    } catch (error) {
      console.error('Process reminder error:', error instanceof Error ? error.message : String(error));
      throw error;
    }
  };

  // Create calendar event function
  const createCalendarEvent = async (title: string, description: string, startTime: Date) => {
    try {
      if (!googleCredentials) {
        throw new Error('Not authenticated with Google');
      }

      // Try to use the server endpoint first for creating events
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        
        const response = await axios.post(
          `${API_BASE_URL}/api/reminders/google-calendar/create-event`,
          {
            credentials: {
              token: googleCredentials.accessToken,
              refresh_token: googleCredentials.refreshToken,
              token_uri: 'https://oauth2.googleapis.com/token',
              client_id: googleCredentials.clientId,
              client_secret: 'GOOGLE_CLIENT_SECRET', // Replace with actual client secret or handle securely
              scopes: ['https://www.googleapis.com/auth/calendar']
            },
            event: {
              summary: title,
              description: description,
              start_time: startTime.toISOString()
            }
          },
          {
            headers: {
              'Authorization': `Bearer ${authToken}`,
              'Content-Type': 'application/json',
            },
          }
        );
        
        return response.data.event;
      } catch (serverError) {
        console.warn('Failed to create event through server, falling back to direct API:', serverError);
        
        // Fall back to direct Google Calendar API if server fails
        const endTime = new Date(startTime);
        endTime.setHours(endTime.getHours() + 1); // Default 1-hour events
        
        const event = {
          summary: title,
          description: description,
          start: {
            dateTime: startTime.toISOString(),
            timeZone: 'Africa/Cairo',
          },
          end: {
            dateTime: endTime.toISOString(),
            timeZone: 'Africa/Cairo',
          },
        };

        const response = await axios.post(
          'https://www.googleapis.com/calendar/v3/calendars/primary/events',
          event,
          {
            headers: {
              Authorization: `Bearer ${googleCredentials.accessToken}`,
              'Content-Type': 'application/json',
            },
          }
        );

        return response.data;
      }
    } catch (error) {
      console.error('Create calendar event error:', error);
      throw error;
    }
  };

  // Get calendar events function
  const getCalendarEvents = async (startDate?: Date, endDate?: Date) => {
    try {
      if (!googleCredentials) {
        throw new Error('Not authenticated with Google');
      }

      // Try to use the server endpoint first for fetching events
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        
        const startParam = startDate ? `start_date=${startDate.toISOString()}` : '';
        const endParam = endDate ? `end_date=${endDate.toISOString()}` : '';
        const queryParams = [startParam, endParam].filter(Boolean).join('&');
        
        const response = await axios.get(
          `${API_BASE_URL}/api/reminders/google-calendar/events?${queryParams}`,
          {
            data: {
              credentials: {
                token: googleCredentials.accessToken,
                refresh_token: googleCredentials.refreshToken,
                token_uri: 'https://oauth2.googleapis.com/token',
                client_id: googleCredentials.clientId,
                client_secret: 'GOOGLE_CLIENT_SECRET', // Replace with actual client secret or handle securely
                scopes: ['https://www.googleapis.com/auth/calendar']
              }
            },
            headers: {
              'Authorization': `Bearer ${authToken}`,
              'Content-Type': 'application/json',
            },
          }
        );
        
        return response.data.events || [];
      } catch (serverError) {
        console.warn('Failed to fetch events through server, falling back to direct API:', serverError);
        
        // Fall back to direct Google Calendar API if server fails
        const now = new Date();
        const timeMin = startDate ? startDate.toISOString() : now.toISOString();
        
        const defaultEnd = new Date();
        defaultEnd.setDate(defaultEnd.getDate() + 7); // Default to 1 week from now
        const timeMax = endDate ? endDate.toISOString() : defaultEnd.toISOString();

        const response = await axios.get(
          'https://www.googleapis.com/calendar/v3/calendars/primary/events',
          {
            params: {
              timeMin,
              timeMax,
              singleEvents: true,
              orderBy: 'startTime',
            },
            headers: {
              Authorization: `Bearer ${googleCredentials.accessToken}`,
            },
          }
        );

        return response.data.items || [];
      }
    } catch (error) {
      console.error('Get calendar events error:', error);
      throw error;
    }
  };

  const value = {
    isAuthenticated,
    signIn: async (email: string, password: string) => {
      try {
        setIsLoading(true);
        setIsAuthenticated(false);
        setUserData(null);
        setGoogleCredentials(null);
        
        const response = await axios.post(`${API_BASE_URL}/api/auth/login`, {
          email,
          password
        });
        
        const { token, user } = response.data;
        
        // Store token and user data
        await AsyncStorage.setItem('authToken', token);
        await AsyncStorage.setItem('userData', JSON.stringify(user));
        
        setIsAuthenticated(true);
        setUserData(user);
        setGoogleCredentials(null);
        
        console.log('Successfully signed in');
        
        // Navigate to the main app
        router.replace('/');
      } catch (error) {
        console.error('Sign in failed:', error instanceof Error ? error.message : String(error));
        setIsAuthenticated(false);
        setUserData(null);
        setGoogleCredentials(null);
      } finally {
        setIsLoading(false);
      }
    },
    signUp: async (name: string, email: string, password: string) => {
      try {
        setIsLoading(true);
        setIsAuthenticated(false);
        setUserData(null);
        setGoogleCredentials(null);
        
        // Store temporary registration data for later use in onboarding
        const tempData = {
          full_name: name,
          email: email,
          password: password
        };
        
        // Store temporary registration data
        await AsyncStorage.setItem('tempRegData', JSON.stringify(tempData));
        setTempRegData(tempData);
        
        console.log('Temporary registration data stored, proceeding to onboarding');
        
        // Navigate to onboarding to collect more user data
        router.push('/onboarding/user-data');
      } catch (error) {
        console.error('Sign up failed:', error instanceof Error ? error.message : String(error));
        setIsAuthenticated(false);
        setUserData(null);
        setGoogleCredentials(null);
      } finally {
        setIsLoading(false);
      }
    },
    completeOnboarding: async (onboardingData: UserDataType) => {
      try {
        setIsLoading(true);
        setIsAuthenticated(false);
        setUserData(null);
        setGoogleCredentials(null);
        
        // Check if we have temporary registration data
        if (!tempRegData) {
          throw new Error('No registration data found. Please sign up first.');
        }
        
        // Prepare complete user data for registration
        const userData = {
          full_name: tempRegData.full_name,
          age: onboardingData.age,
          gender: onboardingData.gender,
          contact_info: {
            email: tempRegData.email,
            phone: onboardingData.phone,
          },
          password: tempRegData.password,
          emergency_contacts: onboardingData.emergency_contacts,
        };
        
        console.log('Registering user with complete data');
        
        // Make API request to register the user
        const response = await axios.post(`${API_BASE_URL}/api/auth/register`, userData);
        
        // Extract the response data
        const responseData = response.data;
        
        // Store user data for our app
        const appUserData = {
          id: responseData.user_id,
          name: tempRegData.full_name,
          email: tempRegData.email,
        };
        
        // Clear temporary registration data
        await AsyncStorage.removeItem('tempRegData');
        setTempRegData(null);
        
        // Store user data and set authenticated
        await AsyncStorage.setItem('userData', JSON.stringify(appUserData));
        await AsyncStorage.setItem('authToken', responseData.token);
        await AsyncStorage.setItem('onboardingCompleted', 'true');
        
        setIsAuthenticated(true);
        setUserData(appUserData);
        setGoogleCredentials(null);
        
        console.log('Registration and onboarding completed successfully');
        
        // Navigate to the main app
        router.replace('/');
      } catch (error) {
        console.error('Onboarding completion failed:', error instanceof Error ? error.message : String(error));
        setIsAuthenticated(false);
        setUserData(null);
        setGoogleCredentials(null);
      } finally {
        setIsLoading(false);
      }
    },
    signOut: async () => {
      try {
        setIsLoading(true);
        console.log('Signing out user...');
        
        // Remove auth token and user data
        await AsyncStorage.removeItem('authToken');
        await AsyncStorage.removeItem('userData');
        await AsyncStorage.removeItem('onboardingCompleted');
        
        setIsAuthenticated(false);
        setUserData(null);
        setGoogleCredentials(null);
        
        console.log('Successfully signed out');
        
        // Navigate to welcome screen
        router.replace('/welcome');
      } catch (error) {
        console.error('Sign out failed:', error instanceof Error ? error.message : String(error));
        setIsAuthenticated(false);
        setUserData(null);
        setGoogleCredentials(null);
      } finally {
        setIsLoading(false);
      }
    },
    fetchUserData: async (userId: string) => {
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        if (!authToken) {
          throw new Error("Authentication token not found");
        }

        const response = await axios.get(`${API_BASE_URL}/api/users/${userId}`, {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        });

        const data = response.data;
        await AsyncStorage.setItem('userData', JSON.stringify(data));
        setUserData(data);
        return data;
      } catch (error) {
        console.error('Error fetching user data:', error instanceof Error ? error.message : String(error));
        throw error;
      }
    },
    updateUserProfile: async (userId: string, userData: any) => {
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        if (!authToken) {
          throw new Error("Authentication token not found");
        }

        const response = await axios.put(`${API_BASE_URL}/api/users/${userId}`, userData, {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        });

        const data = response.data;
        await AsyncStorage.setItem('userData', JSON.stringify(data));
        setUserData(data);
        return data;
      } catch (error) {
        console.error('Error updating user profile:', error instanceof Error ? error.message : String(error));
        throw error;
      }
    },
    fetchMemoryAids: async () => {
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        if (!authToken) {
          throw new Error("Authentication token not found");
        }

        const response = await axios.get(`${API_BASE_URL}/api/memory-aids`, {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        });

        const data = response.data;
        return data;
      } catch (error) {
        console.error('Error fetching memory aids:', error instanceof Error ? error.message : String(error));
        throw error;
      }
    },
    createMemoryAid: async (memoryAidData: MemoryAidDataType) => {
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        if (!authToken) {
          throw new Error("Authentication token not found");
        }

        const response = await axios.post(`${API_BASE_URL}/api/memory-aids`, memoryAidData, {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        });

        const data = response.data;
        return data;
      } catch (error) {
        console.error('Error creating memory aid:', error instanceof Error ? error.message : String(error));
        throw error;
      }
    },
    updateMemoryAid: async (memoryAidId: string, memoryAidData: Partial<MemoryAidDataType>) => {
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        if (!authToken) {
          throw new Error("Authentication token not found");
        }

        const response = await axios.put(`${API_BASE_URL}/api/memory-aids/${memoryAidId}`, memoryAidData, {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        });

        const data = response.data;
        return data;
      } catch (error) {
        console.error('Error updating memory aid:', error instanceof Error ? error.message : String(error));
        throw error;
      }
    },
    deleteMemoryAid: async (memoryAidId: string) => {
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        if (!authToken) {
          throw new Error("Authentication token not found");
        }

        const response = await axios.delete(`${API_BASE_URL}/api/memory-aids/${memoryAidId}`, {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        });

        return true;
      } catch (error) {
        console.error('Error deleting memory aid:', error instanceof Error ? error.message : String(error));
        throw error;
      }
    },
    userData,
    tempRegData,
    googleCredentials,
    googleSignIn,
    googleSignOut,
    createCalendarEvent,
    getCalendarEvents,
    processNaturalLanguageReminder
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Add this default export to fix the error
export default AuthProvider; 