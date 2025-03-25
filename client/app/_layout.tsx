import React, { useEffect, useState } from 'react';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { Provider as PaperProvider, DefaultTheme } from 'react-native-paper';
import { AuthProvider } from './context/AuthContext';
import { View, ActivityIndicator } from 'react-native';
import * as Linking from 'expo-linking';

// Keep the splash screen visible while we fetch resources
SplashScreen.preventAutoHideAsync().catch(() => {
  /* reloading the app might trigger this, ignore if it does */
});

// Define URL scheme for our app
const prefix = Linking.createURL('/');

// Define theme
const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#4285F4',
    accent: '#FF9500',
  },
};

export default function AppLayout() {
  const [appIsReady, setAppIsReady] = useState(false);

  useEffect(() => {
    // Perform any initialization tasks here
    async function prepare() {
      try {
        // Set up deep link handling
        const initialURL = await Linking.getInitialURL();
        console.log('Initial URL:', initialURL);
        
        // Listen for incoming links (while app is running)
        const subscription = Linking.addEventListener('url', handleDeepLink);
        
        // Artificial delay to ensure everything is ready
        await new Promise(resolve => setTimeout(resolve, 500));
        
        return () => {
          // Clean up subscription when component unmounts
          subscription.remove();
        };
      } catch (e) {
        console.warn(e);
      } finally {
        // Tell the application to render
        setAppIsReady(true);
      }
    }

    prepare();
  }, []);
  
  // Handle deep links
  const handleDeepLink = (event: { url: string }) => {
    console.log('Received deep link:', event.url);
    // URL will be handled in the AuthContext
  };

  useEffect(() => {
    if (appIsReady) {
      // This tells the splash screen to hide immediately
      SplashScreen.hideAsync().catch(() => {
        /* ignore error */
      });
    }
  }, [appIsReady]);

  if (!appIsReady) {
    // Show a loading indicator instead of null
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" color="#4285F4" />
      </View>
    );
  }

  return (
    <PaperProvider theme={theme}>
      <AuthProvider>
        <Stack
          screenOptions={{
            headerShown: false,
            animation: 'slide_from_right',
          }}
        >
          <Stack.Screen name="welcome" options={{ gestureEnabled: false }} />
          <Stack.Screen name="index" options={{ gestureEnabled: false }} />
          <Stack.Screen name="auth/signin" />
          <Stack.Screen name="auth/signup" />
          <Stack.Screen name="onboarding/user-data" options={{ gestureEnabled: false }} />
        </Stack>
      </AuthProvider>
    </PaperProvider>
  );
}
