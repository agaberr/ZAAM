import React, { createContext, useState, useContext, useEffect } from 'react';
import { router, useSegments, useRootNavigationState } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Linking from 'expo-linking';
import axios from 'axios';
import { useGoogleAuth, getCurrentUser } from '../services/authService';

// API endpoint base URL
// const API_BASE_URL = 'https://zaam-mj7u.onrender.com'; // For Android emulator pointing to localhost
const API_BASE_URL = 'http://192.168.1.3:5000'; // For Android emulator pointing to localhost

// If using a physical device, use your computer's IP address instead, e.g. 'http://192.168.1.100:5000'


// Define the shape of the auth context
type AuthContextType = {
  isAuthenticated: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  completeOnboarding: (userData: UserDataType) => Promise<void>;
  fetchUserData: (userId: string) => Promise<any>;
  updateUserProfile: (userId: string, userData: any) => Promise<void>;
  fetchMemoryAids: () => Promise<any[]>;
  createMemoryAid: (memoryAidData: MemoryAidDataType) => Promise<any>;
  updateMemoryAid: (memoryAidId: string, memoryAidData: Partial<MemoryAidDataType>) => Promise<any>;
  deleteMemoryAid: (memoryAidId: string) => Promise<boolean>;
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
  signInWithGoogle: async () => {},
  completeOnboarding: async () => {},
  fetchUserData: async () => {},
  updateUserProfile: async () => {},
  fetchMemoryAids: async () => [],
  createMemoryAid: async () => ({}),
  updateMemoryAid: async () => ({}),
  deleteMemoryAid: async () => false,
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
  const [googleAuthPending, setGoogleAuthPending] = useState(false);
  
  const segments = useSegments();
  const navigationState = useRootNavigationState();

  // Add our Google Auth hook
  const { handleGoogleSignIn } = useGoogleAuth();

  // Check if the user is authenticated on initial load
  useEffect(() => {
    const loadAuthState = async () => {
      try {
        console.log('Loading authentication state...');
        const authToken = await AsyncStorage.getItem('authToken');
        const userDataString = await AsyncStorage.getItem('userData');
        const tempRegDataString = await AsyncStorage.getItem('tempRegData');
        
        if (authToken && userDataString) {
          // Parse the user data
          const parsedUserData = JSON.parse(userDataString);
          console.log('Found stored user data:', parsedUserData?.id);
          
          // Validate token by making a simple API request
          try {
            let isValid = false;
            const endpoint = `${API_BASE_URL}/api/users/${parsedUserData.id}`;
            
            console.log('Validating token with endpoint:', endpoint);
            const response = await fetch(endpoint, {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`
              }
            });
            
            if (response.ok) {
              console.log('Token validated successfully');
              isValid = true;
              
              // Update user data with response if available
              try {
                const userData = await response.json();
                if (userData) {
                  // Merge with existing data, keeping all fields
                  const mergedData = {
                    ...parsedUserData,
                    ...userData,
                    // Ensure these important fields aren't overwritten
                    id: parsedUserData.id,
                    token: authToken
                  };
                  
                  // Save the updated data
                  await AsyncStorage.setItem('userData', JSON.stringify(mergedData));
                  setUserData(mergedData);
                } else {
                  setUserData(parsedUserData);
                }
              } catch (parseError) {
                console.error('Error parsing user data response:', parseError);
                setUserData(parsedUserData);
              }
            } else {
              console.log('Token validation failed, response:', response.status);
              isValid = false;
            }
            
            setIsAuthenticated(isValid);
            
            if (!isValid) {
              // Clear invalid token
              await AsyncStorage.removeItem('authToken');
              await AsyncStorage.removeItem('userData');
              setUserData(null);
            }
          } catch (error) {
            console.error('Error validating token:', error);
            // Keep user signed in for offline usage, but they might encounter API errors
            console.log('Keeping user signed in for offline usage');
            setUserData(parsedUserData);
            setIsAuthenticated(true);
          }
        } else {
          console.log('No valid auth data found');
          setIsAuthenticated(false);
          setUserData(null);
        }
        
        if (tempRegDataString) {
          setTempRegData(JSON.parse(tempRegDataString));
        }
      } catch (error) {
        console.error('Failed to load auth state:', error);
        // Clear potentially corrupted data
        AsyncStorage.removeItem('authToken');
        AsyncStorage.removeItem('userData');
        setIsAuthenticated(false);
        setUserData(null);
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

  // Function to handle Google OAuth redirects
  const handleGoogleRedirect = async (url: string) => {
    try {
      console.log('Handling Google redirect:', url);
      setGoogleAuthPending(false);
      
      // Parse the URL to get query parameters
      const urlObj = new URL(url);
      const params = new URLSearchParams(urlObj.search);
      
      // Check if this is a failure redirect
      if (url.includes('google-connect-failure')) {
        const error = params.get('error');
        console.error('Google authentication failed:', error);
        return;
      }
      
      // If it's a success URL for Google sign-in
      if (url.includes('google-auth-success')) {
        const token = params.get('token');
        const user_id = params.get('user_id');
        const is_new = params.get('is_new');
        
        if (!token || !user_id) {
          console.error('Invalid redirect - missing token or user_id');
          return;
        }
        
        // Store the token
        await AsyncStorage.setItem('authToken', token);
        
        // Fetch additional user data
        const userData = {
          id: user_id,
          name: 'Google User', // Placeholder until we fetch more details
          email: '', // Will be filled when we fetch user data
        };
        
        // Store the user data
        await AsyncStorage.setItem('userData', JSON.stringify(userData));
        
        setUserData(userData);
        setIsAuthenticated(true);
        
        console.log('Successfully signed in with Google');
        
        // If this is a new user, redirect to onboarding
        if (is_new === 'true') {
          // Navigate to onboarding
          console.log('New Google user - redirecting to onboarding');
          router.replace('/onboarding/user-data');
        } else {
          // Navigate to the main app
          router.replace('/');
        }
      }
      
      // If it's a success URL for Google connection
      if (url.includes('google-connect-success')) {
        console.log('Successfully connected Google account');
        // This is for existing users linking their Google account
        // Could trigger a refresh of user data here
      }
      
    } catch (error) {
      console.error('Error handling Google redirect:', error);
    }
  };

  // Handle deep links for Google auth callback
  useEffect(() => {
    // Set up URL listener for all deep links, not just when googleAuthPending
    const subscription = Linking.addEventListener('url', (event) => {
      console.log('Received deep link in AuthContext:', event.url);
      
      // Handle Google auth redirects
      if (event.url.includes('google-auth-success') || 
          event.url.includes('google-connect-success') || 
          event.url.includes('google-connect-failure')) {
        handleGoogleRedirect(event.url);
      }
    });
    
    // Check for initial URL that might have launched the app
    const checkInitialURL = async () => {
      const url = await Linking.getInitialURL();
      if (url) {
        console.log('App was launched with URL:', url);
        if (url.includes('google-auth-success') || 
            url.includes('google-connect-success') || 
            url.includes('google-connect-failure')) {
          handleGoogleRedirect(url);
        }
      }
    };
    
    checkInitialURL();
    
    return () => {
      subscription.remove();
    };
  }, []);

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

  // Update the signInWithGoogle function
  const signInWithGoogle = async () => {
    try {
      setGoogleAuthPending(true);
      
      console.log('Starting Google sign-in flow');
      const result = await handleGoogleSignIn();
      
      if (result.success) {
        console.log('Google sign-in successful, setting auth state');
        
        // We don't need to store user data here since it's already 
        // handled by the authService.js saveAuthData function
        
        // Fetch user data to get the full profile
        try {
          const userDataString = await AsyncStorage.getItem('userData');
          if (userDataString) {
            const parsedUserData = JSON.parse(userDataString);
            setUserData(parsedUserData);
            setIsAuthenticated(true);
            
            console.log('Auth state updated in context', parsedUserData);
          }
        } catch (error) {
          console.error('Error parsing saved user data:', error);
        }
        
        if (result.isNew) {
          // Redirect to onboarding for new users
          console.log('New user - redirecting to onboarding');
          router.replace('/onboarding/user-data');
        } else {
          // Navigate to the main app
          console.log('Existing user - redirecting to home');
          router.replace('/');
        }
      } else {
        console.error('Google sign-in failed:', result.error);
      }
    } catch (error) {
      console.error('Error during Google sign-in:', error);
    } finally {
      setGoogleAuthPending(false);
    }
  };

  const value = {
    isAuthenticated,
    signIn,
    signUp,
    signOut,
    signInWithGoogle,
    completeOnboarding,
    fetchUserData: async (userId: string) => {
      try {
        const authToken = await AsyncStorage.getItem('authToken');
        if (!authToken) {
          throw new Error("Authentication token not found");
        }

        const response = await fetch(`${API_BASE_URL}/api/users/${userId}`, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to fetch user data');
        }

        const data = await response.json();
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

        const response = await fetch(`${API_BASE_URL}/api/users/${userId}`, {
          method: 'PUT',
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(userData),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to update user profile');
        }

        const data = await response.json();
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

        const response = await fetch(`${API_BASE_URL}/api/memory-aids`, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to fetch memory aids');
        }

        const data = await response.json();
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

        const response = await fetch(`${API_BASE_URL}/api/memory-aids`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(memoryAidData),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to create memory aid');
        }

        const data = await response.json();
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

        const response = await fetch(`${API_BASE_URL}/api/memory-aids/${memoryAidId}`, {
          method: 'PUT',
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(memoryAidData),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to update memory aid');
        }

        const data = await response.json();
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

        const response = await fetch(`${API_BASE_URL}/api/memory-aids/${memoryAidId}`, {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to delete memory aid');
        }

        return true;
      } catch (error) {
        console.error('Error deleting memory aid:', error instanceof Error ? error.message : String(error));
        throw error;
      }
    },
    userData,
    tempRegData,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Add this default export to fix the error
export default AuthProvider; 