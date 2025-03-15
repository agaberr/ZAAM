import { Stack } from 'expo-router';
import { PaperProvider, MD3LightTheme } from 'react-native-paper';
import { SafeAreaProvider } from 'react-native-safe-area-context';

const theme = {
  ...MD3LightTheme,
  // You can customize your theme colors here
  colors: {
    ...MD3LightTheme.colors,
    primary: '#007AFF', // iOS blue
    secondary: '#5856D6',
  },
};

export default function RootLayout() {
  return (
    <PaperProvider theme={theme}>
      <SafeAreaProvider>
        <Stack>
          <Stack.Screen
            name="index"
            options={{
              headerShown: false,
            }}
          />
          <Stack.Screen
            name="welcome"
            options={{
              headerShown: false,
            }}
          />
          <Stack.Screen
            name="auth/signup"
            options={{
              headerShown: false,
              presentation: 'modal',
            }}
          />
          <Stack.Screen
            name="auth/signin"
            options={{
              headerShown: false,
              presentation: 'modal',
            }}
          />
          <Stack.Screen
            name="onboarding/user-data"
            options={{
              headerShown: false,
              presentation: 'card',
              gestureEnabled: false,
            }}
          />
          <Stack.Screen
            name="dashboard"
            options={{
              headerShown: false,
              gestureEnabled: false,
            }}
          />
          <Stack.Screen
            name="profile"
            options={{
              title: 'Profile',
              presentation: 'modal',
            }}
          />
          <Stack.Screen
            name="talk-to-ai"
            options={{
              title: 'Talk to AI',
              presentation: 'card',
            }}
          />
          <Stack.Screen
            name="reminders"
            options={{
              title: 'Reminders',
              presentation: 'card',
            }}
          />
        </Stack>
      </SafeAreaProvider>
    </PaperProvider>
  );
}
