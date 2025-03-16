import React, { useState, useEffect } from 'react';
import { StyleSheet, View, SafeAreaView, StatusBar } from 'react-native';
import { Provider as PaperProvider, DefaultTheme } from 'react-native-paper';
import { router, useSegments, useRootNavigationState } from 'expo-router';

// Import screens
import HomeScreen from './screens/HomeScreen';
import InsightsScreen from './screens/InsightsScreen';
import TalkToAIScreen from './screens/TalkToAIScreen';
import RemindersScreen from './screens/RemindersScreen';
import ProfileScreen from './screens/ProfileScreen';

// Import components
import BottomNavigation from './components/BottomNavigation';

// Define theme
const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#4285F4',
    accent: '#FF9500',
  },
};

// Auth check function - replace with your actual auth logic
const useProtectedRoute = (isAuthenticated) => {
  const segments = useSegments();
  const navigationState = useRootNavigationState();

  useEffect(() => {
    if (!navigationState?.key) return;
    
    const inAuthGroup = segments[0] === 'auth';
    const inOnboardingGroup = segments[0] === 'onboarding';
    const isWelcomeScreen = segments[0] === 'welcome';
    
    if (
      // If the user is not signed in and the initial segment is not in the auth group
      !isAuthenticated &&
      !inAuthGroup &&
      !inOnboardingGroup &&
      !isWelcomeScreen
    ) {
      // Redirect to the welcome screen
      router.replace('/welcome');
    } else if (isAuthenticated && (inAuthGroup || isWelcomeScreen)) {
      // Redirect away from auth screens if the user is already authenticated
      router.replace('/');
    }
  }, [segments, navigationState?.key, isAuthenticated]);
};

export default function App() {
  // State to track active tab
  const [activeTab, setActiveTab] = useState('home');
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Use the protected route hook
  useProtectedRoute(isAuthenticated);

  // Function to render the active screen
  const renderScreen = () => {
    switch (activeTab) {
      case 'home':
        return <HomeScreen setActiveTab={setActiveTab} />;
      case 'insights':
        return <InsightsScreen setActiveTab={setActiveTab} />;
      case 'ai':
        return <TalkToAIScreen setActiveTab={setActiveTab} />;
      case 'reminders':
        return <RemindersScreen setActiveTab={setActiveTab} />;
      case 'profile':
        return <ProfileScreen setActiveTab={setActiveTab} />;
      default:
        return <HomeScreen setActiveTab={setActiveTab} />;
    }
  };

  return (
    <PaperProvider theme={theme}>
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />
        
        {/* Main content area */}
        <View style={styles.content}>
          {renderScreen()}
        </View>
        
        {/* Bottom Navigation */}
        <BottomNavigation 
          activeTab={activeTab} 
          setActiveTab={setActiveTab} 
        />
      </SafeAreaView>
    </PaperProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  content: {
    flex: 1,
  },
}); 