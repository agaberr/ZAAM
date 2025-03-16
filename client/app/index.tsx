import React, { useState } from 'react';
import { StyleSheet, View, SafeAreaView, StatusBar } from 'react-native';
import { useAuth } from './context/AuthContext';

// Import screens from your screens directory
import HomeScreen from './screens/HomeScreen';
import InsightsScreen from './screens/InsightsScreen';
import TalkToAIScreen from './screens/TalkToAIScreen';
import RemindersScreen from './screens/RemindersScreen';
import ProfileScreen from './screens/ProfileScreen';

// Import components
import BottomNavigation from './components/BottomNavigation';

export default function App() {
  // State to track active tab
  const [activeTab, setActiveTab] = useState('home');
  const { isAuthenticated } = useAuth();

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