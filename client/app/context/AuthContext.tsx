import React, { createContext, useState, useContext, useEffect } from 'react';
import { router, useSegments, useRootNavigationState } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Mock user data
const MOCK_USERS = [
  {
    id: '1',
    name: 'Ahmed',
    email: 'admin',
    password: 'admin',
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
  completeOnboarding: () => Promise<void>;
  userData: any | null;
};

// Create the context with a default value
const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  signIn: async () => {},
  signUp: async () => {},
  signOut: async () => {},
  completeOnboarding: async () => {},
  userData: null,
});

// Custom hook to use the auth context
export const useAuth = () => useContext(AuthContext);

// Provider component that wraps your app and makes auth object available to any child component
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [userData, setUserData] = useState<any | null>(null);
  
  const segments = useSegments();
  const navigationState = useRootNavigationState();

  // Check if the user is authenticated on initial load
  useEffect(() => {
    const loadAuthState = async () => {
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        const userDataString = await AsyncStorage.getItem('userData');
        
        if (authToken && userDataString) {
          const parsedUserData = JSON.parse(userDataString);
          setUserData(parsedUserData);
          setIsAuthenticated(true);
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
    if (isLoading || !navigationState?.key) return;
    
    const inAuthGroup = segments[0] === 'auth';
    const inOnboardingGroup = segments[0] === 'onboarding';
    const isWelcomeScreen = segments[0] === 'welcome' || segments.length === 0;
    
    if (!isAuthenticated && !inAuthGroup && !inOnboardingGroup && !isWelcomeScreen) {
      // Redirect to the welcome screen if not authenticated
      router.replace('/welcome');
    } else if (isAuthenticated && (inAuthGroup || isWelcomeScreen)) {
      // Redirect to the home screen if authenticated and on auth screens
      router.replace('/');
    }
  }, [isAuthenticated, segments, navigationState?.key, isLoading]);

  // Sign in function with mock data
  const signIn = async (email: string, password: string) => {
    try {
      // Check against mock users
      const user = MOCK_USERS.find(
        u => (u.email.toLowerCase() === email.toLowerCase() || u.name.toLowerCase() === email.toLowerCase()) && 
             u.password === password
      );
      
      if (!user) {
        throw new Error('Invalid credentials');
      }
      
      // Store user data and token
      const userData = {
        id: user.id,
        name: user.name,
        email: user.email,
      };
      
      await AsyncStorage.setItem('authToken', 'mock-auth-token-' + user.id);
      await AsyncStorage.setItem('userData', JSON.stringify(userData));
      
      setUserData(userData);
      setIsAuthenticated(true);
      
      console.log('Successfully signed in as:', user.name);
      
      // Navigate to the main app
      router.replace('/');
    } catch (error) {
      console.error('Sign in failed:', error);
      throw error;
    }
  };

  // Sign up function
  const signUp = async (name: string, email: string, password: string) => {
    try {
      // In a real app, you would register the user with your backend
      // For now, we'll just create a mock user
      const newUser = {
        id: String(MOCK_USERS.length + 1),
        name,
        email,
        password,
      };
      
      // Store user data and token
      const userData = {
        id: newUser.id,
        name: newUser.name,
        email: newUser.email,
      };
      
      await AsyncStorage.setItem('authToken', 'mock-auth-token-' + newUser.id);
      await AsyncStorage.setItem('userData', JSON.stringify(userData));
      
      setUserData(userData);
      setIsAuthenticated(true);
      
      console.log('Successfully signed up as:', newUser.name);
      
      // Navigate to onboarding
      router.push('/onboarding/user-data');
    } catch (error) {
      console.error('Sign up failed:', error);
      throw error;
    }
  };

  // Complete onboarding function
  const completeOnboarding = async () => {
    try {
      // In a real app, you would save onboarding data to your backend
      await AsyncStorage.setItem('onboardingCompleted', 'true');
      
      console.log('Onboarding completed');
      
      // Navigate to the main app
      router.replace('/');
    } catch (error) {
      console.error('Onboarding completion failed:', error);
      throw error;
    }
  };

  // Sign out function
  const signOut = async () => {
    try {
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
      console.error('Sign out failed:', error);
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
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
} 