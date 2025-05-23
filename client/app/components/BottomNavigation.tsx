import React from "react";
import { StyleSheet, View, TouchableOpacity, Text } from "react-native";
import { Ionicons } from "@expo/vector-icons";

interface BottomNavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

// Define the type for navigation items
interface NavItem {
  id: string;
  label: string;
  icon: "home" | "analytics" | "chatbubble-ellipses" | "calendar" | "person";
}

export default function BottomNavigation({
  activeTab,
  setActiveTab,
}: BottomNavigationProps) {
  // Navigation items
  const navItems: NavItem[] = [
    { id: "reminders", label: "Reminders", icon: "calendar" },
    { id: "ai", label: "AI", icon: "chatbubble-ellipses" },
    { id: "profile", label: "Profile", icon: "person" },
  ];

  return (
    <View style={styles.container}>
      {navItems.map((item) => (
        <TouchableOpacity
          key={item.id}
          style={styles.navItem}
          onPress={() => setActiveTab(item.id)}
        >
          <Ionicons
            name={activeTab === item.id ? item.icon : `${item.icon}-outline`}
            size={24}
            color={activeTab === item.id ? "#4285F4" : "#8E8E93"}
          />
          <Text
            style={[
              styles.navLabel,
              { color: activeTab === item.id ? "#4285F4" : "#8E8E93" },
            ]}
          >
            {item.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    height: 60,
    backgroundColor: "#fff",
    borderTopWidth: 0,
    paddingHorizontal: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 8,
  },
  navItem: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  navLabel: {
    fontSize: 12,
    marginTop: 4,
  },
});
