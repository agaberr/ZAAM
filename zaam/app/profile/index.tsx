import React, { useState } from 'react';
import { StyleSheet, View, ScrollView } from 'react-native';
import { Text, Surface, List, Switch, Button, Avatar, IconButton, TextInput, Divider } from 'react-native-paper';
import { router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

export default function ProfileScreen() {
  const [editMode, setEditMode] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [darkMode, setDarkMode] = useState(false);

  // User data state
  const [userData, setUserData] = useState({
    name: 'John Doe',
    email: 'john.doe@example.com',
    gender: 'Male',
    age: '65',
    medications: 'Aspirin 100mg\nVitamin D 1000 IU',
    medicalInfo: 'Mild cognitive impairment\nHigh blood pressure',
    caregiverName: 'Jane Doe',
    caregiverPhone: '+1 234-567-8900',
  });

  const handleSave = () => {
    setEditMode(false);
    // TODO: Save changes to backend
  };

  const handleLogout = () => {
    // TODO: Implement logout logic
    router.replace('/welcome');
  };

  return (
    <View style={styles.container}>
      <StatusBar style="light" />

      {/* Header */}
      <Surface style={styles.header} elevation={2}>
        <View style={styles.headerContent}>
          <IconButton icon="arrow-left" size={24} onPress={() => router.back()} />
          <Text variant="titleLarge" style={styles.title}>Profile & Settings</Text>
          <IconButton
            icon={editMode ? 'check' : 'pencil'}
            size={24}
            onPress={() => editMode ? handleSave() : setEditMode(true)}
          />
        </View>
      </Surface>

      <ScrollView style={styles.content}>
        {/* Profile Section */}
        <View style={styles.profileSection}>
          <Avatar.Icon size={80} icon="account" style={styles.avatar} />
          <Text variant="headlineSmall" style={styles.name}>{userData.name}</Text>
          <Text variant="bodyLarge" style={styles.email}>{userData.email}</Text>
        </View>

        <List.Section>
          <List.Subheader>Personal Information</List.Subheader>
          
          <TextInput
            label="Full Name"
            value={userData.name}
            onChangeText={(text) => setUserData({ ...userData, name: text })}
            mode="outlined"
            disabled={!editMode}
            style={styles.input}
          />

          <TextInput
            label="Age"
            value={userData.age}
            onChangeText={(text) => setUserData({ ...userData, age: text })}
            mode="outlined"
            disabled={!editMode}
            keyboardType="numeric"
            style={styles.input}
          />

          <TextInput
            label="Gender"
            value={userData.gender}
            onChangeText={(text) => setUserData({ ...userData, gender: text })}
            mode="outlined"
            disabled={!editMode}
            style={styles.input}
          />

          <TextInput
            label="Medications"
            value={userData.medications}
            onChangeText={(text) => setUserData({ ...userData, medications: text })}
            mode="outlined"
            disabled={!editMode}
            multiline
            numberOfLines={3}
            style={styles.input}
          />

          <TextInput
            label="Medical Information"
            value={userData.medicalInfo}
            onChangeText={(text) => setUserData({ ...userData, medicalInfo: text })}
            mode="outlined"
            disabled={!editMode}
            multiline
            numberOfLines={3}
            style={styles.input}
          />
        </List.Section>

        <Divider />

        <List.Section>
          <List.Subheader>Caregiver Information</List.Subheader>
          
          <TextInput
            label="Caregiver's Name"
            value={userData.caregiverName}
            onChangeText={(text) => setUserData({ ...userData, caregiverName: text })}
            mode="outlined"
            disabled={!editMode}
            style={styles.input}
          />

          <TextInput
            label="Caregiver's Phone"
            value={userData.caregiverPhone}
            onChangeText={(text) => setUserData({ ...userData, caregiverPhone: text })}
            mode="outlined"
            disabled={!editMode}
            style={styles.input}
          />
        </List.Section>

        <Divider />

        <List.Section>
          <List.Subheader>Preferences</List.Subheader>
          
          <List.Item
            title="Notifications"
            right={() => (
              <Switch
                value={notificationsEnabled}
                onValueChange={setNotificationsEnabled}
              />
            )}
          />

          <List.Item
            title="Voice Interaction"
            right={() => (
              <Switch
                value={voiceEnabled}
                onValueChange={setVoiceEnabled}
              />
            )}
          />

          <List.Item
            title="Dark Mode"
            right={() => (
              <Switch
                value={darkMode}
                onValueChange={setDarkMode}
              />
            )}
          />
        </List.Section>

        <Divider />

        <List.Section>
          <List.Subheader>Account</List.Subheader>
          
          <List.Item
            title="Change Password"
            left={props => <List.Icon {...props} icon="lock" />}
            onPress={() => console.log('Change password')}
          />

          <List.Item
            title="Privacy Policy"
            left={props => <List.Icon {...props} icon="shield-account" />}
            onPress={() => console.log('Privacy policy')}
          />

          <List.Item
            title="Terms of Service"
            left={props => <List.Icon {...props} icon="file-document" />}
            onPress={() => console.log('Terms of service')}
          />
        </List.Section>

        <Button
          mode="contained"
          onPress={handleLogout}
          style={styles.logoutButton}
          icon="logout"
        >
          Log Out
        </Button>

        {/* Bottom Spacing */}
        <View style={styles.bottomSpacing} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#007AFF',
    paddingTop: 60,
    paddingBottom: 20,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
  },
  title: {
    flex: 1,
    color: '#fff',
    marginLeft: 16,
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
  },
  profileSection: {
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#fff',
  },
  avatar: {
    backgroundColor: '#007AFF',
    marginBottom: 12,
  },
  name: {
    fontWeight: 'bold',
    marginBottom: 4,
  },
  email: {
    opacity: 0.7,
  },
  input: {
    marginHorizontal: 16,
    marginBottom: 12,
    backgroundColor: '#fff',
  },
  logoutButton: {
    margin: 16,
    backgroundColor: '#FF3B30',
  },
  bottomSpacing: {
    height: 20,
  },
}); 