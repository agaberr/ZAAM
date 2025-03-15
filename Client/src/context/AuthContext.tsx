import React, { createContext, useState, useEffect, useContext } from 'react';
import * as SecureStore from 'expo-secure-store';
import axios from 'axios';

// Define the User type based on our schema
interface User {
  _id: string;
  full_name: string;
  age: number;
  gender: string;
  contact_info: {
    phone: string;
    email: string;
  };
  emergency_contacts: Array<{
    name: string;
    relationship: string;
    phone: string;
  }>;
  preferences: {
    language: string;
    voice_type: string;
    reminder_frequency: string;
  };
  created_at: string;
  updated_at: string;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (userData: Partial<User>, password: string) => Promise<void>;
  logout: () => Promise<void>;
  updateUser: (userData: Partial<User>) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Mock API URL - replace with your actual backend URL
const API_URL = 'https://your-backend-api.com';

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check if user is already logged in
  useEffect(() => {
    const loadUser = async () => {
      try {
        const token = await SecureStore.getItemAsync('userToken');
        if (token) {
          // Configure axios with the auth token
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          
          // In a real app, you would fetch the user profile from your API
          // const response = await axios.get(`${API_URL}/api/users/profile`);
          // setUser(response.data);
          
          // For demo purposes, we'll use mock data
          setUser({
            _id: '123456789',
            full_name: 'John Doe',
            age: 75,
            gender: 'Male',
            contact_info: {
              phone: '+1234567890',
              email: 'johndoe@example.com',
            },
            emergency_contacts: [
              {
                name: 'Jane Doe',
                relationship: 'Daughter',
                phone: '+9876543210',
              },
            ],
            preferences: {
              language: 'English',
              voice_type: 'Male',
              reminder_frequency: 'hourly',
            },
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
          
          setIsAuthenticated(true);
        }
      } catch (error) {
        console.error('Failed to load user', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadUser();
  }, []);

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    try {
      // In a real app, you would make an API call to authenticate
      // const response = await axios.post(`${API_URL}/api/auth/login`, { email, password });
      // const { token, user } = response.data;
      
      // For demo purposes, we'll use mock data
      const token = 'mock-jwt-token';
      const mockUser: User = {
        _id: '123456789',
        full_name: 'John Doe',
        age: 75,
        gender: 'Male',
        contact_info: {
          phone: '+1234567890',
          email: email,
        },
        emergency_contacts: [
          {
            name: 'Jane Doe',
            relationship: 'Daughter',
            phone: '+9876543210',
          },
        ],
        preferences: {
          language: 'English',
          voice_type: 'Male',
          reminder_frequency: 'hourly',
        },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      
      // Save the token
      await SecureStore.setItemAsync('userToken', token);
      
      // Set the authorization header for future requests
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      setUser(mockUser);
      setIsAuthenticated(true);
    } catch (error) {
      console.error('Login failed', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const signup = async (userData: Partial<User>, password: string) => {
    setIsLoading(true);
    try {
      // In a real app, you would make an API call to register
      // const response = await axios.post(`${API_URL}/api/auth/register`, { ...userData, password });
      // const { token, user } = response.data;
      
      // For demo purposes, we'll use mock data
      const token = 'mock-jwt-token';
      const mockUser: User = {
        _id: '123456789',
        full_name: userData.full_name || 'New User',
        age: userData.age || 70,
        gender: userData.gender || 'Unspecified',
        contact_info: {
          phone: userData.contact_info?.phone || '',
          email: userData.contact_info?.email || '',
        },
        emergency_contacts: userData.emergency_contacts || [],
        preferences: {
          language: 'English',
          voice_type: 'Female',
          reminder_frequency: 'daily',
        },
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      
      // Save the token
      await SecureStore.setItemAsync('userToken', token);
      
      // Set the authorization header for future requests
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      setUser(mockUser);
      setIsAuthenticated(true);
    } catch (error) {
      console.error('Signup failed', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    try {
      // Remove the token
      await SecureStore.deleteItemAsync('userToken');
      
      // Remove the authorization header
      delete axios.defaults.headers.common['Authorization'];
      
      setUser(null);
      setIsAuthenticated(false);
    } catch (error) {
      console.error('Logout failed', error);
    }
  };

  const updateUser = async (userData: Partial<User>) => {
    try {
      // In a real app, you would make an API call to update the user
      // const response = await axios.put(`${API_URL}/api/users/profile`, userData);
      // setUser(response.data);
      
      // For demo purposes, we'll update the local state
      if (user) {
        const updatedUser = { ...user, ...userData, updated_at: new Date().toISOString() };
        setUser(updatedUser);
      }
    } catch (error) {
      console.error('Update user failed', error);
      throw error;
    }
  };

  return (
    <AuthContext.Provider value={{ 
      user, 
      isLoading, 
      isAuthenticated, 
      login, 
      signup, 
      logout, 
      updateUser 
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
