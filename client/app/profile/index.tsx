import React, { useState } from 'react';
import { StyleSheet, View, ScrollView } from 'react-native';
import { Text, Card, Avatar, Switch, IconButton, Button, useTheme, Divider } from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';

type AppRoute = '/dashboard' | '/reminders' | '/talk-to-ai' | '/profile';

export default function ProfileScreen() {
  const theme = useTheme();
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);

  const handleNavigation = (route: AppRoute) => {
    router.push(route as any); // TODO: Update route types when proper type definitions are available
  };

  const handleLogout = () => {
    router.replace('/');
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <IconButton icon="arrow-left" onPress={() => router.back()} />
          <Text variant="titleLarge">Profile & Settings</Text>
        </View>
        <IconButton icon="cog" />
      </View>

      <ScrollView style={styles.content}>
        {/* Profile Section */}
        <Card style={styles.profileCard}>
          <Card.Content style={styles.profileContent}>
            <View style={styles.profileHeader}>
              <Avatar.Image
                size={80}
                source={require('../assets/default-avatar.png')}
              />
              <View style={styles.profileInfo}>
                <Text variant="headlineSmall" style={styles.name}>Noah Turner</Text>
                <Text variant="bodyLarge" style={styles.email}>noah.turner@example.com</Text>
                <Button
                  mode="outlined"
                  icon="pencil"
                  style={styles.editButton}
                  labelStyle={styles.editButtonLabel}
                >
                  Edit Profile
                </Button>
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* Health Information */}
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Text variant="titleMedium" style={styles.sectionTitle}>Health Information</Text>
            <View style={styles.infoRow}>
              <View style={styles.infoItem}>
                <Text variant="bodyMedium" style={styles.infoLabel}>Age</Text>
                <Text variant="titleMedium">65</Text>
              </View>
              <View style={styles.infoItem}>
                <Text variant="bodyMedium" style={styles.infoLabel}>Gender</Text>
                <Text variant="titleMedium">Male</Text>
              </View>
              <View style={styles.infoItem}>
                <Text variant="bodyMedium" style={styles.infoLabel}>Blood Type</Text>
                <Text variant="titleMedium">A+</Text>
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* Caregiver Information */}
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Text variant="titleMedium" style={styles.sectionTitle}>Caregiver Contact</Text>
            <View style={styles.caregiverInfo}>
              <Avatar.Icon size={40} icon="account" style={styles.caregiverIcon} />
              <View style={styles.caregiverDetails}>
                <Text variant="titleMedium">Sarah Turner</Text>
                <Text variant="bodyMedium" style={styles.relationship}>Daughter</Text>
                <Text variant="bodyMedium">+1 (555) 123-4567</Text>
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* App Settings */}
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Text variant="titleMedium" style={styles.sectionTitle}>App Settings</Text>
            
            <View style={styles.settingRow}>
              <View style={styles.settingInfo}>
                <IconButton icon="bell" size={24} />
                <Text variant="bodyLarge">Notifications</Text>
              </View>
              <Switch value={notificationsEnabled} onValueChange={setNotificationsEnabled} />
            </View>
            
            <Divider style={styles.divider} />
            
            <View style={styles.settingRow}>
              <View style={styles.settingInfo}>
                <IconButton icon="theme-light-dark" size={24} />
                <Text variant="bodyLarge">Dark Mode</Text>
              </View>
              <Switch value={darkMode} onValueChange={setDarkMode} />
            </View>
            
            <Divider style={styles.divider} />
            
            <View style={styles.settingRow}>
              <View style={styles.settingInfo}>
                <IconButton icon="microphone" size={24} />
                <Text variant="bodyLarge">Voice Interaction</Text>
              </View>
              <Switch value={voiceEnabled} onValueChange={setVoiceEnabled} />
            </View>
          </Card.Content>
        </Card>

        {/* Account Actions */}
        <View style={styles.accountActions}>
          <Button
            mode="contained"
            icon="logout"
            onPress={handleLogout}
            style={styles.logoutButton}
          >
            Log Out
          </Button>
          <Button
            mode="text"
            textColor="red"
            style={styles.deleteButton}
          >
            Delete Account
          </Button>
        </View>
      </ScrollView>

      {/* Bottom Navigation */}
      <Card style={styles.bottomNav}>
        <Card.Content style={styles.bottomNavContent}>
          <IconButton 
            icon="home" 
            size={24}
            onPress={() => handleNavigation('/dashboard')}
          />
          <IconButton 
            icon="calendar" 
            size={24}
            onPress={() => handleNavigation('/reminders')}
          />
          <IconButton 
            icon="robot" 
            size={24}
            onPress={() => handleNavigation('/talk-to-ai')}
          />
          <IconButton
            icon="cog"
            size={24}
            mode="contained"
            containerColor="#E8F5F1"
            iconColor="#00B383"
            onPress={() => handleNavigation('/profile')}
          />
        </Card.Content>
      </Card>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F6FA',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 4,
    paddingVertical: 8,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  profileCard: {
    marginBottom: 16,
    borderRadius: 16,
  },
  profileContent: {
    paddingVertical: 24,
  },
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  profileInfo: {
    flex: 1,
    gap: 4,
  },
  name: {
    fontWeight: 'bold',
  },
  email: {
    color: '#666',
    marginBottom: 8,
  },
  editButton: {
    borderRadius: 20,
    borderColor: '#2D68FF',
    alignSelf: 'flex-start',
  },
  editButtonLabel: {
    color: '#2D68FF',
  },
  sectionCard: {
    marginBottom: 16,
    borderRadius: 16,
  },
  sectionTitle: {
    fontWeight: 'bold',
    marginBottom: 16,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  infoItem: {
    alignItems: 'center',
    gap: 4,
  },
  infoLabel: {
    color: '#666',
  },
  caregiverInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  caregiverIcon: {
    backgroundColor: '#E8F5F1',
  },
  caregiverDetails: {
    gap: 4,
  },
  relationship: {
    color: '#666',
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  settingInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  divider: {
    marginVertical: 8,
  },
  accountActions: {
    gap: 8,
    marginTop: 8,
    marginBottom: 32,
  },
  logoutButton: {
    borderRadius: 12,
  },
  deleteButton: {
    borderRadius: 12,
  },
  bottomNav: {
    borderRadius: 0,
    elevation: 8,
  },
  bottomNavContent: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 0,
  },
}); 