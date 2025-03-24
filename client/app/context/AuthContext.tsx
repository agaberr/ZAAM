import React, { createContext, useState, useContext, useEffect } from 'react';
import { router, useSegments, useRootNavigationState } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';

// API endpoint base URL
const API_BASE_URL = 'http://localhost:5000'; // For Android emulator pointing to localhost
// If using a physical device, use your computer's IP address instead, e.g. 'http://192.168.1.100:5000'

// Mock user data
const MOCK_USERS = [
  {
    id: '1',
    name: 'Ahmed',
    email: 'ahmed',
    password: 'ahmed',
  },
  {
    id: '2',
    name: 'Test User',
    email: 'test@example.com',
    password: 'password123',
  }
];

// Define the shape of the auth context
type AuthContextType = {
  isAuthenticated: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  completeOnboarding: (userData: UserDataType) => Promise<void>;
  userData: any | null;
  tempRegData: TempRegDataType | null;
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

// Create the context with a default value
const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  signIn: async () => {},
  signUp: async () => {},
  signOut: async () => {},
  completeOnboarding: async () => {},
  userData: null,
  tempRegData: null,
});

// Custom hook to use the auth context
export const useAuth = () => useContext(AuthContext);

// Provider component that wraps your app and makes auth object available to any child component
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [userData, setUserData] = useState<any | null>(null);
  const [tempRegData, setTempRegData] = useState<TempRegDataType | null>(null);
  
  const segments = useSegments();
  const navigationState = useRootNavigationState();

  // Check if the user is authenticated on initial load
  useEffect(() => {
    const loadAuthState = async () => {
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        const userDataString = await AsyncStorage.getItem('userData');
        const tempRegDataString = await AsyncStorage.getItem('tempRegData');
        
        if (authToken && userDataString) {
          const parsedUserData = JSON.parse(userDataString);
          setUserData(parsedUserData);
          setIsAuthenticated(true);
        }
        
        if (tempRegDataString) {
          setTempRegData(JSON.parse(tempRegDataString));
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

  // Sign in function with API integration
  const signIn = async (email: string, password: string) => {
    try {
      // Prepare data for API request
      const loginData = {
        email: email,
        password: password
      };

      // Make API request to log in
      const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(loginData),
      });

      // Check if request was successful
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Login failed');
      }

      // Extract the response data
      const responseData = await response.json();
      
      // Store token
      await AsyncStorage.setItem('authToken', responseData.token);
      
      // Fetch user data or use what was returned
      // For now, we'll use what we have
      const userData = {
        id: responseData.user_id,
        // The backend doesn't currently return the user's name, 
        // so we'll need to fetch it separately or handle it differently
        name: 'User', // Placeholder
        email: email,
      };
      
      await AsyncStorage.setItem('userData', JSON.stringify(userData));
      
      setUserData(userData);
      setIsAuthenticated(true);
      
      console.log('Successfully signed in');
      
      // Navigate to the main app
      router.replace('/');
    } catch (error) {
      console.error('Sign in failed:', error instanceof Error ? error.message : String(error));
      throw error;
    }
  };

  // Sign up function
  const signUp = async (name: string, email: string, password: string) => {
    try {
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
      throw error;
    }
  };

  // Complete onboarding function
  const completeOnboarding = async (onboardingData: UserDataType) => {
    try {
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
      const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      // Check if request was successful
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Registration failed');
      }

      // Extract the response data
      const responseData = await response.json();
      
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
      await AsyncStorage.setItem('onboardingCompleted', 'true');
      
      setUserData(appUserData);
      setIsAuthenticated(true);
      
      console.log('Registration and onboarding completed successfully');
      
      // Navigate to the main app
      router.replace('/');
    } catch (error) {
      console.error('Onboarding completion failed:', error instanceof Error ? error.message : String(error));
      throw error;
    }
  };

  // Sign out function
  const signOut = async () => {
    try {
      console.log('Signing out user...');
      
      // Remove auth token and user data
      await AsyncStorage.removeItem('authToken');
      await AsyncStorage.removeItem('userData');
      await AsyncStorage.removeItem('onboardingCompleted');
      
      setIsAuthenticated(false);
      setUserData(null);
      
      console.log('Successfully signed out');
      
      // Navigate to welcome screen
      router.replace('/welcome');
    } catch (error) {
      console.error('Sign out failed:', error instanceof Error ? error.message : String(error));
      throw error;
    }
  };

  const value = {
    isAuthenticated,
    signIn,
    signUp,
    signOut,
    completeOnboarding,
    userData,
    tempRegData,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Add this default export to fix the error
export default AuthProvider; 