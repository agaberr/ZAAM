import React, { useEffect, useState } from 'react';
import { Stack, SplashScreen } from 'expo-router';
import { Provider as PaperProvider, DefaultTheme } from 'react-native-paper';
import { AuthProvider } from './context/AuthContext';

// Prevent the splash screen from auto-hiding
SplashScreen.preventAutoHideAsync().catch(() => {
  /* ignore error */
});

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
        // Artificial delay to ensure everything is ready
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (e) {
        console.warn(e);
      } finally {
        // Tell the application to render
        setAppIsReady(true);
      }
    }

    prepare();
  }, []);

  useEffect(() => {
    if (appIsReady) {
      // This tells the splash screen to hide immediately
      SplashScreen.hideAsync().catch(() => {
        /* ignore error */
      });
    }
  }, [appIsReady]);

  if (!appIsReady) {
    return null;
  }

  return (
    <PaperProvider theme={theme}>
      <AuthProvider>
        <Stack
          screenOptions={{
            headerShown: false,
            animation: 'slide_from_right',
          }}
          initialRouteName="welcome"
        >
          <Stack.Screen name="welcome" options={{ gestureEnabled: false }} />
          <Stack.Screen name="index" options={{ gestureEnabled: false }} />
          <Stack.Screen name="auth/signin" />
          <Stack.Screen name="auth/signup" />
          <Stack.Screen name="onboarding/user-data" />
        </Stack>
      </AuthProvider>
    </PaperProvider>
  );
}
