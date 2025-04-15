import React, { useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { View, StyleSheet, ActivityIndicator } from 'react-native';
import LoginScreen from './LoginScreen';
import { getCurrentUser } from '../services/authService';
import { router } from 'expo-router';

export default function GoogleAuthScreen() {
  useEffect(() => {
    // Check if user is already authenticated on screen load
    const checkAuthStatus = async () => {
      try {
        console.log('Checking authentication status...');
        const user = await getCurrentUser();
        
        if (user) {
          console.log('User is already authenticated, redirecting to home');
          // User is already authenticated, redirect to home
          router.replace('/');
        }
      } catch (error) {
        console.error('Error checking auth status:', error);
        // If there's an error, we'll stay on this screen to let the user retry
      }
    };
    
    checkAuthStatus();
  }, []);

  return (
    <View style={styles.container}>
      <StatusBar style="auto" />
      <LoginScreen />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
}); 