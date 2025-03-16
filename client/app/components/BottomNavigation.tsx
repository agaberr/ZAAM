import React from 'react';
import { StyleSheet, View, TouchableOpacity, Text } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface BottomNavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export default function BottomNavigation({ activeTab, setActiveTab }: BottomNavigationProps) {
  // Navigation items
  const navItems = [
    { id: 'home', label: 'Home', icon: 'home' },
    { id: 'insights', label: 'Insights', icon: 'analytics' },
    { id: 'ai', label: 'AI', icon: 'chatbubble-ellipses' },
    { id: 'reminders', label: 'Reminders', icon: 'calendar' },
    { id: 'profile', label: 'Profile', icon: 'person' },
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
            color={activeTab === item.id ? '#4285F4' : '#8E8E93'}
          />
          <Text
            style={[
              styles.navLabel,
              { color: activeTab === item.id ? '#4285F4' : '#8E8E93' },
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
    flexDirection: 'row',
    height: 60,
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#E1E1E1',
    paddingHorizontal: 10,
  },
  navItem: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  navLabel: {
    fontSize: 12,
    marginTop: 4,
  },
}); 