import { useState } from 'react';
import { Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as WebBrowser from 'expo-web-browser';
import Constants from 'expo-constants';

// API URL and Google Client ID from constants
const API_URL = Constants.expoConfig?.extra?.apiUrl || 'http://localhost:5000/api';
const GOOGLE_CLIENT_ID = Constants.expoConfig?.extra?.googleWebClientId || '';

// For debugging
console.log('API URL:', API_URL);
console.log('Google Client ID:', GOOGLE_CLIENT_ID);

// Remove the WebBrowser completion call since we're using a different approach
// WebBrowser.maybeCompleteAuthSession();

export const useGoogleAuth = () => {
  const [loading, setLoading] = useState(false);

  const handleGoogleSignIn = async () => {
    try {
      setLoading(true);
      
      // First, get the authorization URL from the backend
      const authResponse = await fetch(`${API_URL}/auth/google/connect`);
      const authData = await authResponse.json();
      
      if (!authData.authorization_url) {
        throw new Error('Failed to get authorization URL');
      }
      
      console.log('Opening auth URL:', authData.authorization_url);
      
      // Open browser for OAuth flow
      const result = await WebBrowser.openAuthSessionAsync(
        authData.authorization_url,
        'zaam://callback'
      );
      
      console.log('Auth result:', result);
      
      if (result.type === 'success') {
        // Extract data from the redirect URL
        const url = new URL(result.url);
        const token = url.searchParams.get('token');
        const userId = url.searchParams.get('user_id');
        
        console.log('Got token:', token ? 'Yes' : 'No');
        console.log('Got user_id:', userId);
        
        if (token && userId) {
          // We already have the token and user_id directly from the URL
          const data = {
            success: true,
            token: token,
            user_id: userId,
            is_new: false  // Assuming not a new user, update if needed
          };
          
          // Save authentication data
          await saveAuthData(data);
          
          // After successful authentication, fetch user profile data
          try {
            await fetchAndSaveUserProfile(token, userId);
          } catch (profileError) {
            console.error('Error fetching user profile:', profileError);
            // Continue even if profile fetch fails
          }
          
          return {
            success: true,
            token: token,
            userId: userId,
            isNew: data.is_new
          };
        } else {
          // If we don't have token and userId, try the old code-based flow
          const code = url.searchParams.get('code');
          const state = url.searchParams.get('state');
          
          console.log('Got code:', code ? 'Yes' : 'No');
          console.log('Got state:', state);
          
          if (!code) {
            throw new Error('No authorization data received');
          }
          
          // Exchange the code for tokens with our backend
          const tokenResponse = await fetch(`${API_URL}/auth/google/callback`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              code,
              state: state || authData.state
            })
          });

          const data = await tokenResponse.json();
          console.log('Token response:', data);
          
          if (data.success) {
            // Save authentication data
            await saveAuthData(data);
            
            // After successful authentication, fetch user profile data if not a new user
            if (!data.is_new) {
              try {
                await fetchAndSaveUserProfile(data.token, data.user_id);
              } catch (profileError) {
                console.error('Error fetching user profile:', profileError);
                // Continue even if profile fetch fails
              }
            }
            
            return {
              success: true,
              token: data.token,
              userId: data.user_id,
              isNew: data.is_new
            };
          } else {
            throw new Error(data.error || 'Authentication failed');
          }
        }
      } else if (result.type === 'cancel') {
        throw new Error('Google authentication was cancelled');
      } else {
        throw new Error('Google authentication failed');
      }
    } catch (error) {
      console.error('Google sign-in error:', error);
      return {
        success: false,
        error: error.message
      };
    } finally {
      setLoading(false);
    }
  };

  return {
    handleGoogleSignIn,
    isLoading: loading
  };
};

// Helper function to save authentication data
const saveAuthData = async (data) => {
  try {
    // Store the auth token and user info
    await AsyncStorage.setItem('authToken', data.token);
    await AsyncStorage.setItem('userId', data.user_id);
    
    // Create basic user data object to store
    const userData = {
      id: data.user_id,
      token: data.token,
      isNewUser: data.is_new,
      authMethod: 'google',
      timestamp: new Date().toISOString()
    };
    
    // Store user data
    await AsyncStorage.setItem('userData', JSON.stringify(userData));
    
    // Store new user flag separately
    if (data.is_new) {
      await AsyncStorage.setItem('isNewUser', 'true');
    } else {
      await AsyncStorage.removeItem('isNewUser');
    }
    
    console.log('Authentication data saved successfully');
    return true;
  } catch (error) {
    console.error('Error saving auth data:', error);
    throw error;
  }
};

// Helper function to fetch and save user profile
const fetchAndSaveUserProfile = async (token, userId) => {
  try {
    const response = await fetch(`${API_URL}/users/${userId}`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (!response.ok) {
      // Try with the /api prefix if the first request fails
      const apiResponse = await fetch(`${API_URL.includes('/api') ? API_URL : `${API_URL}/api`}/users/${userId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (apiResponse.ok) {
        const profileData = await apiResponse.json();
        await saveProfileData(profileData, userId);
        return;
      }
      
      throw new Error(`Failed to fetch user profile: ${response.status}`);
    }
    
    const profileData = await response.json();
    await saveProfileData(profileData, userId);
  } catch (error) {
    console.error('Error fetching user profile:', error);
    throw error;
  }
};

// Helper function to save profile data to AsyncStorage
const saveProfileData = async (profileData, userId) => {
  try {
    // Update the stored user data with profile info
    const userDataStr = await AsyncStorage.getItem('userData');
    if (userDataStr) {
      const userData = JSON.parse(userDataStr);
      const updatedUserData = {
        ...userData,
        name: profileData.name || profileData.full_name,
        email: profileData.email,
        picture: profileData.profile_picture,
        profile: profileData
      };
      
      await AsyncStorage.setItem('userData', JSON.stringify(updatedUserData));
      console.log('User profile data saved successfully');
    }
  } catch (error) {
    console.error('Error saving profile data:', error);
    throw error;
  }
};

export const signOut = async () => {
  try {
    // Get the token first before it's removed
    const token = await AsyncStorage.getItem('authToken');
    
    // Clear stored auth data
    await AsyncStorage.removeItem('authToken');
    await AsyncStorage.removeItem('userId');
    await AsyncStorage.removeItem('isNewUser');
    
    // Call backend to revoke tokens
    if (token) {
      await fetch(`${API_URL}/auth/google/disconnect`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
    }
    
    return { success: true };
  } catch (error) {
    console.error('Sign out error:', error);
    return { success: false, error: error.message };
  }
};

export const getCurrentUser = async () => {
  try {
    const token = await AsyncStorage.getItem('authToken');
    const userId = await AsyncStorage.getItem('userId');
    
    if (!token || !userId) {
      return null;
    }
    
    // Verify token with backend
    const response = await fetch(`${API_URL}/auth/google/status`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (response.ok) {
      return {
        token,
        userId,
        isNew: await AsyncStorage.getItem('isNewUser') === 'true'
      };
    }
    
    return null;
  } catch (error) {
    console.error('Get current user error:', error);
    return null;
  }
}; 