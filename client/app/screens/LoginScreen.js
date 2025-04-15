import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import { useGoogleAuth, getCurrentUser } from '../services/authService';
import { useNavigation, useRouter } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { router } from 'expo-router';

export default function LoginScreen() {
  const [error, setError] = useState(null);
  const { handleGoogleSignIn, isLoading } = useGoogleAuth();
  const navigation = useNavigation();

  useEffect(() => {
    // Check if user is already logged in
    const checkAuth = async () => {
      try {
        const user = await getCurrentUser();
        if (user) {
          navigateAfterAuth(user.isNew);
        }
      } catch (err) {
        console.error('Error checking auth:', err);
      }
    };
    checkAuth();
  }, []);

  // Function to handle navigation after authentication
  const navigateAfterAuth = (isNewUser) => {
    try {
      console.log('Navigating after auth, new user:', isNewUser);
      
      // If using Expo Router
      if (router) {
        if (isNewUser) {
          router.replace('/onboarding/user-data');
        } else {
          router.replace('/');
        }
        return;
      }
      
      // Fallback to React Navigation if Expo Router is not available
      if (navigation) {
        if (isNewUser) {
          navigation.reset({
            index: 0,
            routes: [{ name: 'Onboarding' }],
          });
        } else {
          navigation.reset({
            index: 0,
            routes: [{ name: 'Home' }],
          });
        }
      }
    } catch (err) {
      console.error('Navigation error:', err);
      Alert.alert('Navigation Error', 'There was a problem navigating to the next screen.');
    }
  };

  const handleSignIn = async () => {
    try {
      setError(null);
      
      const result = await handleGoogleSignIn();
      console.log('Google sign-in result:', result);
      
      if (result.success) {
        // Authentication data is already saved by the auth service
        console.log('Authentication successful, navigating to appropriate screen');
        navigateAfterAuth(result.isNew);
      } else {
        setError(result.error || 'Authentication failed');
      }
    } catch (err) {
      console.error('Sign in error:', err);
      setError(err.message || 'An unexpected error occurred');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome to ZAAM</Text>
      <Text style={styles.subtitle}>Your Alzheimer's AI Companion</Text>
      
      {error && (
        <Text style={styles.error}>{error}</Text>
      )}
      
      <TouchableOpacity
        style={styles.button}
        onPress={handleSignIn}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" size="small" />
        ) : (
          <Text style={styles.buttonText}>Sign in with Google</Text>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#fff'
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#6200ee'
  },
  subtitle: {
    fontSize: 18,
    color: '#666',
    marginBottom: 40,
    textAlign: 'center'
  },
  button: {
    backgroundColor: '#4285F4',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 200
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 10
  },
  error: {
    color: 'red',
    marginBottom: 20,
    textAlign: 'center'
  }
}); 