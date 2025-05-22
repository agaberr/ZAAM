import React, { useState, useEffect } from "react";
import { StyleSheet, View, SafeAreaView, StatusBar } from "react-native";
import { useAuth } from "./context/AuthContext";
import { router, useRootNavigationState } from "expo-router";

// Import screens from your screens directory
import HomeScreen from "./screens/HomeScreen";
import TalkToAIScreen from "./screens/TalkToAIScreen";
import RemindersScreen from "./screens/RemindersScreen";
import ProfileScreen from "./screens/ProfileScreen";
import ThreeAvatar from "./screens/ThreeScreen";

// Import components
import BottomNavigation from "./components/BottomNavigation";

export default function App() {
  // State to track active tab
  const [activeTab, setActiveTab] = useState("home");
  const { isAuthenticated } = useAuth();
  const rootNavigationState = useRootNavigationState();

  // Check authentication on mount, but only after root layout is mounted
  useEffect(() => {
    if (!rootNavigationState?.key) return; // Return early if navigation state isn't ready

    if (!isAuthenticated) {
      router.replace("/welcome");
    }
  }, [isAuthenticated, rootNavigationState?.key]);

  // Function to render the active screen
  const renderScreen = () => {
    switch (activeTab) {
      case "home":
        return <HomeScreen setActiveTab={setActiveTab} />;
      case "ai":
        return <ThreeAvatar setActiveTab={setActiveTab} />;
      case "reminders":
        return <RemindersScreen setActiveTab={setActiveTab} />;
      case "profile":
        return <ProfileScreen setActiveTab={setActiveTab} />;
      default:
        return <HomeScreen setActiveTab={setActiveTab} />;
    }
  };

  // If not authenticated, show a loading state instead of immediate redirect
  if (!isAuthenticated && rootNavigationState?.key) {
    return null;
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />

      {/* Main content area */}
      <View style={styles.content}>{renderScreen()}</View>

      {/* Bottom Navigation */}
      <BottomNavigation activeTab={activeTab} setActiveTab={setActiveTab} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FFFFFF",
  },
  content: {
    flex: 1,
  },
});
