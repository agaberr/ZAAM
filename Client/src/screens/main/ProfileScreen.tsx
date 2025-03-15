import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, Alert } from 'react-native';
import { 
  Avatar, 
  Title, 
  Text, 
  Card, 
  List, 
  Switch, 
  Button, 
  Divider,
  TextInput,
  Dialog,
  Portal,
  Chip
} from 'react-native-paper';
import { useAuth } from '../../context/AuthContext';

const ProfileScreen = () => {
  const { user, logout, updateUser } = useAuth();
  
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  
  const [editDialogVisible, setEditDialogVisible] = useState(false);
  const [editField, setEditField] = useState('');
  const [editValue, setEditValue] = useState('');
  const [editTitle, setEditTitle] = useState('');
  
  const [emergencyContactDialogVisible, setEmergencyContactDialogVisible] = useState(false);
  const [contactName, setContactName] = useState('');
  const [contactRelationship, setContactRelationship] = useState('');
  const [contactPhone, setContactPhone] = useState('');

  const handleLogout = () => {
    Alert.alert(
      'Confirm Logout',
      'Are you sure you want to log out?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Logout',
          onPress: () => logout(),
          style: 'destructive',
        },
      ]
    );
  };

  const openEditDialog = (field, value, title) => {
    setEditField(field);
    setEditValue(value);
    setEditTitle(title);
    setEditDialogVisible(true);
  };

  const handleSaveEdit = async () => {
    try {
      // Prepare the update object based on the field being edited
      let updateData = {};
      
      if (editField === 'full_name') {
        updateData = { full_name: editValue };
      } else if (editField === 'age') {
        updateData = { age: parseInt(editValue) };
      } else if (editField === 'gender') {
        updateData = { gender: editValue };
      } else if (editField === 'phone') {
        updateData = { 
          contact_info: { 
            ...user.contact_info, 
            phone: editValue 
          } 
        };
      } else if (editField === 'email') {
        updateData = { 
          contact_info: { 
            ...user.contact_info, 
            email: editValue 
          } 
        };
      } else if (editField === 'language') {
        updateData = { 
          preferences: { 
            ...user.preferences, 
            language: editValue 
          } 
        };
      } else if (editField === 'voice_type') {
        updateData = { 
          preferences: { 
            ...user.preferences, 
            voice_type: editValue 
          } 
        };
      } else if (editField === 'reminder_frequency') {
        updateData = { 
          preferences: { 
            ...user.preferences, 
            reminder_frequency: editValue 
          } 
        };
      }
      
      await updateUser(updateData);
      setEditDialogVisible(false);
    } catch (error) {
      console.error('Error updating profile', error);
      Alert.alert('Update Failed', 'Failed to update profile information.');
    }
  };

  const handleAddEmergencyContact = async () => {
    if (!contactName || !contactRelationship || !contactPhone) {
      Alert.alert('Missing Information', 'Please fill in all fields for the emergency contact.');
      return;
    }
    
    try {
      const newContact = {
        name: contactName,
        relationship: contactRelationship,
        phone: contactPhone,
      };
      
      const updatedContacts = [...(user.emergency_contacts || []), newContact];
      
      await updateUser({ emergency_contacts: updatedContacts });
      
      // Reset form
      setContactName('');
      setContactRelationship('');
      setContactPhone('');
      setEmergencyContactDialogVisible(false);
    } catch (error) {
      console.error('Error adding emergency contact', error);
      Alert.alert('Update Failed', 'Failed to add emergency contact.');
    }
  };

  const handleRemoveEmergencyContact = async (index) => {
    try {
      const updatedContacts = [...user.emergency_contacts];
      updatedContacts.splice(index, 1);
      
      await updateUser({ emergency_contacts: updatedContacts });
    } catch (error) {
      console.error('Error removing emergency contact', error);
      Alert.alert('Update Failed', 'Failed to remove emergency contact.');
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Avatar.Text 
          size={80} 
          label={user?.full_name?.split(' ').map(n => n[0]).join('') || 'U'} 
          style={styles.avatar}
        />
        <Title style={styles.name}>{user?.full_name || 'User'}</Title>
        <Text style={styles.subtitle}>Patient Profile</Text>
      </View>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.sectionTitle}>Personal Information</Title>
          
          <List.Item
            title="Full Name"
            description={user?.full_name || 'Not set'}
            left={props => <List.Icon {...props} icon="account" />}
            right={props => <List.Icon {...props} icon="pencil" />}
            onPress={() => openEditDialog('full_name', user?.full_name || '', 'Edit Full Name')}
          />
          <Divider />
          
          <List.Item
            title="Age"
            description={user?.age?.toString() || 'Not set'}
            left={props => <List.Icon {...props} icon="calendar" />}
            right={props => <List.Icon {...props} icon="pencil" />}
            onPress={() => openEditDialog('age', user?.age?.toString() || '', 'Edit Age')}
          />
          <Divider />
          
          <List.Item
            title="Gender"
            description={user?.gender || 'Not set'}
            left={props => <List.Icon {...props} icon="gender-male-female" />}
            right={props => <List.Icon {...props} icon="pencil" />}
            onPress={() => openEditDialog('gender', user?.gender || '', 'Edit Gender')}
          />
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.sectionTitle}>Contact Information</Title>
          
          <List.Item
            title="Phone"
            description={user?.contact_info?.phone || 'Not set'}
            left={props => <List.Icon {...props} icon="phone" />}
            right={props => <List.Icon {...props} icon="pencil" />}
            onPress={() => openEditDialog('phone', user?.contact_info?.phone || '', 'Edit Phone')}
          />
          <Divider />
          
          <List.Item
            title="Email"
            description={user?.contact_info?.email || 'Not set'}
            left={props => <List.Icon {...props} icon="email" />}
            right={props => <List.Icon {...props} icon="pencil" />}
            onPress={() => openEditDialog('email', user?.contact_info?.email || '', 'Edit Email')}
          />
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <View style={styles.sectionHeader}>
            <Title style={styles.sectionTitle}>Emergency Contacts</Title>
            <Button 
              mode="text" 
              onPress={() => setEmergencyContactDialogVisible(true)}
              style={styles.addButton}
            >
              Add
            </Button>
          </View>
          
          {user?.emergency_contacts?.length > 0 ? (
            user.emergency_contacts.map((contact, index) => (
              <View key={index}>
                <List.Item
                  title={contact.name}
                  description={`${contact.relationship} • ${contact.phone}`}
                  left={props => <List.Icon {...props} icon="account-alert" />}
                  right={props => (
                    <Button 
                      icon="delete" 
                      mode="text" 
                      onPress={() => handleRemoveEmergencyContact(index)}
                      style={styles.deleteButton}
                    />
                  )}
                />
                {index < user.emergency_contacts.length - 1 && <Divider />}
              </View>
            ))
          ) : (
            <Text style={styles.emptyText}>No emergency contacts added</Text>
          )}
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.sectionTitle}>Preferences</Title>
          
          <List.Item
            title="Language"
            description={user?.preferences?.language || 'English'}
            left={props => <List.Icon {...props} icon="translate" />}
            right={props => <List.Icon {...props} icon="pencil" />}
            onPress={() => openEditDialog('language', user?.preferences?.language || 'English', 'Edit Language')}
          />
          <Divider />
          
          <List.Item
            title="Voice Type"
            description={user?.preferences?.voice_type || 'Female'}
            left={props => <List.Icon {...props} icon="account-voice" />}
            right={props => <List.Icon {...props} icon="pencil" />}
            onPress={() => openEditDialog('voice_type', user?.preferences?.voice_type || 'Female', 'Edit Voice Type')}
          />
          <Divider />
          
          <List.Item
            title="Reminder Frequency"
            description={user?.preferences?.reminder_frequency || 'Daily'}
            left={props => <List.Icon {...props} icon="bell" />}
            right={props => <List.Icon {...props} icon="pencil" />}
            onPress={() => openEditDialog('reminder_frequency', user?.preferences?.reminder_frequency || 'Daily', 'Edit Reminder Frequency')}
          />
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.sectionTitle}>App Settings</Title>
          
          <List.Item
            title="Notifications"
            left={props => <List.Icon {...props} icon="bell-outline" />}
            right={() => (
              <Switch
                value={notificationsEnabled}
                onValueChange={setNotificationsEnabled}
                color="#6200ee"
              />
            )}
          />
          <Divider />
          
          <List.Item
            title="Voice Responses"
            left={props => <List.Icon {...props} icon="volume-high" />}
            right={() => (
              <Switch
                value={voiceEnabled}
                onValueChange={setVoiceEnabled}
                color="#6200ee"
              />
            )}
          />
          <Divider />
          
          <List.Item
            title="Dark Mode"
            left={props => <List.Icon {...props} icon="theme-light-dark" />}
            right={() => (
              <Switch
                value={darkMode}
                onValueChange={setDarkMode}
                color="#6200ee"
              />
            )}
          />
        </Card.Content>
      </Card>

      <Button 
        mode="contained" 
        onPress={handleLogout}
        style={styles.logoutButton}
        icon="logout"
      >
        Log Out
      </Button>

      <View style={styles.spacer} />

      <Portal>
        <Dialog visible={editDialogVisible} onDismiss={() => setEditDialogVisible(false)}>
          <Dialog.Title>{editTitle}</Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Value"
              value={editValue}
              onChangeText={setEditValue}
              mode="outlined"
              style={styles.dialogInput}
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setEditDialogVisible(false)}>Cancel</Button>
            <Button onPress={handleSaveEdit}>Save</Button>
          </Dialog.Actions>
        </Dialog>

        <Dialog 
          visible={emergencyContactDialogVisible} 
          onDismiss={() => setEmergencyContactDialogVisible(false)}
        >
          <Dialog.Title>Add Emergency Contact</Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Name"
              value={contactName}
              onChangeText={setContactName}
              mode="outlined"
              style={styles.dialogInput}
            />
            <TextInput
              label="Relationship"
              value={contactRelationship}
              onChangeText={setContactRelationship}
              mode="outlined"
              style={styles.dialogInput}
            />
            <TextInput
              label="Phone Number"
              value={contactPhone}
              onChangeText={setContactPhone}
              keyboardType="phone-pad"
              mode="outlined"
              style={styles.dialogInput}
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setEmergencyContactDialogVisible(false)}>Cancel</Button>
            <Button onPress={handleAddEmergencyContact}>Add</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#6200ee',
  },
  avatar: {
    backgroundColor: '#ffffff',
  },
  name: {
    color: '#ffffff',
    marginTop: 10,
    fontSize: 24,
  },
  subtitle: {
    color: '#ffffff',
    opacity: 0.8,
  },
  card: {
    margin: 16,
    marginTop: 8,
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 18,
    marginBottom: 10,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  addButton: {
    marginTop: -5,
  },
  deleteButton: {
    marginVertical: -8,
  },
  emptyText: {
    textAlign: 'center',
    fontStyle: 'italic',
    marginTop: 10,
    marginBottom: 10,
    color: '#757575',
  },
  logoutButton: {
    margin: 16,
    marginTop: 8,
    backgroundColor: '#f44336',
  },
  spacer: {
    height: 20,
  },
  dialogInput: {
    marginBottom: 10,
  },
});

export default ProfileScreen;
