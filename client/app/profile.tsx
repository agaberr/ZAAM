import React, { useState } from 'react';
import { StyleSheet, View, ScrollView, Image, TouchableOpacity, Switch, Alert } from 'react-native';
import { Text, Button, Card, Avatar, TextInput, Divider, List, IconButton, Chip } from 'react-native-paper';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import BottomNavigation from './components/BottomNavigation';

interface Medication {
  id: string;
  name: string;
  dosage: string;
  frequency: string;
  time: string;
}

interface EmergencyContact {
  id: string;
  name: string;
  relationship: string;
  phone: string;
  isCaregiver: boolean;
}

interface MemoryAid {
  id: string;
  title: string;
  description: string;
  date: string;
  type: 'person' | 'place' | 'event' | 'object';
}

export default function ProfileScreen() {
  const router = useRouter();
  
  // User profile state
  const [name, setName] = useState('Amit Kumar');
  const [age, setAge] = useState('75');
  const [gender, setGender] = useState('Male');
  const [email, setEmail] = useState('amit.kumar@example.com');
  const [phone, setPhone] = useState('+1 (555) 123-4567');
  const [profileImage, setProfileImage] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  
  // Preferences state
  const [voiceType, setVoiceType] = useState('Male');
  const [language, setLanguage] = useState('English');
  const [reminderFrequency, setReminderFrequency] = useState('Hourly');
  const [textSize, setTextSize] = useState('Medium');
  const [highContrastMode, setHighContrastMode] = useState(false);
  const [voiceCommandsEnabled, setVoiceCommandsEnabled] = useState(true);
  const [locationTrackingEnabled, setLocationTrackingEnabled] = useState(true);
  const [nightModeEnabled, setNightModeEnabled] = useState(false);
  
  // Medications state
  const [medications, setMedications] = useState<Medication[]>([
    { id: '1', name: 'Aspirin', dosage: '100mg', frequency: 'Daily', time: '9:00 AM' },
    { id: '2', name: 'Donepezil', dosage: '5mg', frequency: 'Daily', time: '1:00 PM' },
    { id: '3', name: 'Memantine', dosage: '10mg', frequency: 'Twice daily', time: '9:00 AM, 9:00 PM' }
  ]);
  
  // Emergency contacts state
  const [emergencyContacts, setEmergencyContacts] = useState<EmergencyContact[]>([
    { id: '1', name: 'Sarah Kumar', relationship: 'Daughter', phone: '+1 (555) 987-6543', isCaregiver: true },
    { id: '2', name: 'Raj Kumar', relationship: 'Son', phone: '+1 (555) 456-7890', isCaregiver: false },
    { id: '3', name: 'Dr. Smith', relationship: 'Physician', phone: '+1 (555) 234-5678', isCaregiver: false }
  ]);
  
  // Memory aids state (creative addition)
  const [memoryAids, setMemoryAids] = useState<MemoryAid[]>([
    { 
      id: '1', 
      title: 'Sarah', 
      description: 'Your daughter who visits every Sunday. She has two children named Maya and Rohan.', 
      date: '2023-05-15',
      type: 'person'
    },
    { 
      id: '2', 
      title: 'Home Address', 
      description: '123 Maple Street, Apt 4B, Springfield, IL 62704', 
      date: '2023-05-15',
      type: 'place'
    },
    { 
      id: '3', 
      title: 'Wedding Anniversary', 
      description: 'You were married on June 12, 1970 to Priya at the Grand Temple.', 
      date: '2023-05-20',
      type: 'event'
    }
  ]);
  
  // Activity stats (creative addition)
  const [activityStats, setActivityStats] = useState({
    daysActive: 24,
    medicationAdherence: 92,
    aiInteractions: 78,
    reminderCompletion: 85
  });
  
  // Sections expanded state
  const [expandedSections, setExpandedSections] = useState({
    personalInfo: true,
    preferences: false,
    medications: false,
    emergencyContacts: false,
    memoryAids: false,
    account: false
  });
  
  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections({
      ...expandedSections,
      [section]: !expandedSections[section]
    });
  };
  
  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled) {
      setProfileImage(result.assets[0].uri);
    }
  };
  
  const handleLogout = () => {
    Alert.alert(
      "Log Out",
      "Are you sure you want to log out?",
      [
        {
          text: "Cancel",
          style: "cancel"
        },
        { 
          text: "Log Out", 
          onPress: () => router.replace('/welcome'),
          style: "destructive"
        }
      ]
    );
  };
  
  const handleDeleteAccount = () => {
    Alert.alert(
      "Delete Account",
      "Are you sure you want to delete your account? This action cannot be undone.",
      [
        {
          text: "Cancel",
          style: "cancel"
        },
        { 
          text: "Delete", 
          onPress: () => {
            // In a real app, this would call an API to delete the account
            router.replace('/welcome');
          },
          style: "destructive"
        }
      ]
    );
  };
  
  const renderMemoryAidIcon = (type: string) => {
    switch(type) {
      case 'person':
        return <Ionicons name="person" size={24} color="#4285F4" />;
      case 'place':
        return <Ionicons name="location" size={24} color="#FF9500" />;
      case 'event':
        return <Ionicons name="calendar" size={24} color="#34C759" />;
      case 'object':
        return <Ionicons name="cube" size={24} color="#FF3B30" />;
      default:
        return <Ionicons name="help-circle" size={24} color="#8E8E93" />;
    }
  };
  
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <IconButton
          icon="arrow-left"
          size={24}
          onPress={() => router.back()}
        />
        <Text style={styles.headerTitle}>Profile</Text>
        <Button 
          mode="text" 
          onPress={() => setEditMode(!editMode)}
          labelStyle={{color: '#4285F4'}}
        >
          {editMode ? 'Save' : 'Edit'}
        </Button>
      </View>
      
      <ScrollView style={styles.scrollView}>
        {/* Profile Header with Image and Stats */}
        <View style={styles.profileHeader}>
          <TouchableOpacity 
            style={styles.profileImageContainer}
            onPress={editMode ? pickImage : undefined}
            disabled={!editMode}
          >
            {profileImage ? (
              <Image source={{ uri: profileImage }} style={styles.profileImage} />
            ) : (
              <Avatar.Text 
                size={100} 
                label={name.split(' ').map(n => n[0]).join('')} 
                style={styles.profileAvatar}
              />
            )}
            {editMode && (
              <View style={styles.editImageOverlay}>
                <Ionicons name="camera" size={24} color="white" />
              </View>
            )}
          </TouchableOpacity>
          
          <Text style={styles.profileName}>{name}</Text>
          
          {/* Activity Stats Cards */}
          <View style={styles.statsContainer}>
            <Card style={styles.statCard}>
              <Card.Content style={styles.statCardContent}>
                <Text style={styles.statValue}>{activityStats.daysActive}</Text>
                <Text style={styles.statLabel}>Days Active</Text>
              </Card.Content>
            </Card>
            
            <Card style={styles.statCard}>
              <Card.Content style={styles.statCardContent}>
                <Text style={styles.statValue}>{activityStats.medicationAdherence}%</Text>
                <Text style={styles.statLabel}>Med Adherence</Text>
              </Card.Content>
            </Card>
            
            <Card style={styles.statCard}>
              <Card.Content style={styles.statCardContent}>
                <Text style={styles.statValue}>{activityStats.aiInteractions}</Text>
                <Text style={styles.statLabel}>AI Interactions</Text>
              </Card.Content>
            </Card>
          </View>
          
          {/* Wellness Score (creative addition) */}
          <Card style={styles.wellnessCard}>
            <Card.Content>
              <View style={styles.wellnessHeader}>
                <Text style={styles.wellnessTitle}>Wellness Score</Text>
                <Chip mode="outlined" style={styles.wellnessChip}>Good</Chip>
              </View>
              
              <View style={styles.progressBarContainer}>
                <View style={[styles.progressBar, { width: `${activityStats.reminderCompletion}%` }]} />
              </View>
              
              <Text style={styles.wellnessDescription}>
                Based on your medication adherence, activity level, and AI interactions
              </Text>
            </Card.Content>
          </Card>
        </View>
        
        {/* Personal Information Section */}
        <List.Accordion
          title="Personal Information"
          left={props => <List.Icon {...props} icon="account" />}
          expanded={expandedSections.personalInfo}
          onPress={() => toggleSection('personalInfo')}
          style={styles.accordion}
        >
          <Card style={styles.sectionCard}>
            <Card.Content>
              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>Full Name</Text>
                {editMode ? (
                  <TextInput
                    value={name}
                    onChangeText={setName}
                    style={styles.textInput}
                    mode="outlined"
                    dense
                  />
                ) : (
                  <Text style={styles.inputValue}>{name}</Text>
                )}
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>Age</Text>
                {editMode ? (
                  <TextInput
                    value={age}
                    onChangeText={setAge}
                    style={styles.textInput}
                    mode="outlined"
                    dense
                    keyboardType="number-pad"
                  />
                ) : (
                  <Text style={styles.inputValue}>{age}</Text>
                )}
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>Gender</Text>
                {editMode ? (
                  <TextInput
                    value={gender}
                    onChangeText={setGender}
                    style={styles.textInput}
                    mode="outlined"
                    dense
                  />
                ) : (
                  <Text style={styles.inputValue}>{gender}</Text>
                )}
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>Email</Text>
                {editMode ? (
                  <TextInput
                    value={email}
                    onChangeText={setEmail}
                    style={styles.textInput}
                    mode="outlined"
                    dense
                    keyboardType="email-address"
                  />
                ) : (
                  <Text style={styles.inputValue}>{email}</Text>
                )}
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>Phone</Text>
                {editMode ? (
                  <TextInput
                    value={phone}
                    onChangeText={setPhone}
                    style={styles.textInput}
                    mode="outlined"
                    dense
                    keyboardType="phone-pad"
                  />
                ) : (
                  <Text style={styles.inputValue}>{phone}</Text>
                )}
              </View>
            </Card.Content>
          </Card>
        </List.Accordion>
        
        {/* AI Interaction Preferences */}
        <List.Accordion
          title="AI Interaction Preferences"
          left={props => <List.Icon {...props} icon="robot" />}
          expanded={expandedSections.preferences}
          onPress={() => toggleSection('preferences')}
          style={styles.accordion}
        >
          <Card style={styles.sectionCard}>
            <Card.Content>
              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>AI Voice Type</Text>
                {editMode ? (
                  <View style={styles.chipContainer}>
                    <Chip 
                      selected={voiceType === 'Male'} 
                      onPress={() => setVoiceType('Male')}
                      style={styles.selectionChip}
                    >
                      Male
                    </Chip>
                    <Chip 
                      selected={voiceType === 'Female'} 
                      onPress={() => setVoiceType('Female')}
                      style={styles.selectionChip}
                    >
                      Female
                    </Chip>
                  </View>
                ) : (
                  <Text style={styles.inputValue}>{voiceType}</Text>
                )}
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>Language</Text>
                {editMode ? (
                  <View style={styles.chipContainer}>
                    <Chip 
                      selected={language === 'English'} 
                      onPress={() => setLanguage('English')}
                      style={styles.selectionChip}
                    >
                      English
                    </Chip>
                    <Chip 
                      selected={language === 'Hindi'} 
                      onPress={() => setLanguage('Hindi')}
                      style={styles.selectionChip}
                    >
                      Hindi
                    </Chip>
                    <Chip 
                      selected={language === 'Spanish'} 
                      onPress={() => setLanguage('Spanish')}
                      style={styles.selectionChip}
                    >
                      Spanish
                    </Chip>
                  </View>
                ) : (
                  <Text style={styles.inputValue}>{language}</Text>
                )}
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>Reminder Frequency</Text>
                {editMode ? (
                  <View style={styles.chipContainer}>
                    <Chip 
                      selected={reminderFrequency === 'Hourly'} 
                      onPress={() => setReminderFrequency('Hourly')}
                      style={styles.selectionChip}
                    >
                      Hourly
                    </Chip>
                    <Chip 
                      selected={reminderFrequency === 'Daily'} 
                      onPress={() => setReminderFrequency('Daily')}
                      style={styles.selectionChip}
                    >
                      Daily
                    </Chip>
                    <Chip 
                      selected={reminderFrequency === 'As Needed'} 
                      onPress={() => setReminderFrequency('As Needed')}
                      style={styles.selectionChip}
                    >
                      As Needed
                    </Chip>
                  </View>
                ) : (
                  <Text style={styles.inputValue}>{reminderFrequency}</Text>
                )}
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.inputRow}>
                <Text style={styles.inputLabel}>Text Size</Text>
                {editMode ? (
                  <View style={styles.chipContainer}>
                    <Chip 
                      selected={textSize === 'Small'} 
                      onPress={() => setTextSize('Small')}
                      style={styles.selectionChip}
                    >
                      Small
                    </Chip>
                    <Chip 
                      selected={textSize === 'Medium'} 
                      onPress={() => setTextSize('Medium')}
                      style={styles.selectionChip}
                    >
                      Medium
                    </Chip>
                    <Chip 
                      selected={textSize === 'Large'} 
                      onPress={() => setTextSize('Large')}
                      style={styles.selectionChip}
                    >
                      Large
                    </Chip>
                  </View>
                ) : (
                  <Text style={styles.inputValue}>{textSize}</Text>
                )}
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.switchRow}>
                <Text style={styles.switchLabel}>High Contrast Mode</Text>
                <Switch
                  value={highContrastMode}
                  onValueChange={setHighContrastMode}
                  disabled={!editMode}
                  trackColor={{ false: '#D1D1D6', true: '#A2C5FF' }}
                  thumbColor={highContrastMode ? '#4285F4' : '#F4F4F4'}
                />
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.switchRow}>
                <Text style={styles.switchLabel}>Voice Commands</Text>
                <Switch
                  value={voiceCommandsEnabled}
                  onValueChange={setVoiceCommandsEnabled}
                  disabled={!editMode}
                  trackColor={{ false: '#D1D1D6', true: '#A2C5FF' }}
                  thumbColor={voiceCommandsEnabled ? '#4285F4' : '#F4F4F4'}
                />
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.switchRow}>
                <Text style={styles.switchLabel}>Location Tracking</Text>
                <Switch
                  value={locationTrackingEnabled}
                  onValueChange={setLocationTrackingEnabled}
                  disabled={!editMode}
                  trackColor={{ false: '#D1D1D6', true: '#A2C5FF' }}
                  thumbColor={locationTrackingEnabled ? '#4285F4' : '#F4F4F4'}
                />
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.switchRow}>
                <Text style={styles.switchLabel}>Night Mode</Text>
                <Switch
                  value={nightModeEnabled}
                  onValueChange={setNightModeEnabled}
                  disabled={!editMode}
                  trackColor={{ false: '#D1D1D6', true: '#A2C5FF' }}
                  thumbColor={nightModeEnabled ? '#4285F4' : '#F4F4F4'}
                />
              </View>
            </Card.Content>
          </Card>
        </List.Accordion>
        
        {/* Medications */}
        <List.Accordion
          title="Medications"
          left={props => <List.Icon {...props} icon="pill" />}
          expanded={expandedSections.medications}
          onPress={() => toggleSection('medications')}
          style={styles.accordion}
        >
          <Card style={styles.sectionCard}>
            <Card.Content>
              {medications.map((medication, index) => (
                <React.Fragment key={medication.id}>
                  <View style={styles.medicationItem}>
                    <View style={styles.medicationIcon}>
                      <MaterialCommunityIcons name="pill" size={24} color="#4285F4" />
                    </View>
                    <View style={styles.medicationDetails}>
                      <Text style={styles.medicationName}>{medication.name}</Text>
                      <Text style={styles.medicationInfo}>
                        {medication.dosage} • {medication.frequency} • {medication.time}
                      </Text>
                    </View>
                    {editMode && (
                      <IconButton
                        icon="delete"
                        size={20}
                        onPress={() => {
                          const updatedMedications = [...medications];
                          updatedMedications.splice(index, 1);
                          setMedications(updatedMedications);
                        }}
                        style={styles.deleteButton}
                      />
                    )}
                  </View>
                  {index < medications.length - 1 && <Divider style={styles.divider} />}
                </React.Fragment>
              ))}
              
              {editMode && (
                <Button 
                  mode="outlined" 
                  icon="plus" 
                  onPress={() => {
                    // In a real app, this would open a form to add a new medication
                    const newMed: Medication = {
                      id: `${medications.length + 1}`,
                      name: 'New Medication',
                      dosage: 'Dosage',
                      frequency: 'Frequency',
                      time: 'Time'
                    };
                    setMedications([...medications, newMed]);
                  }}
                  style={styles.addButton}
                >
                  Add Medication
                </Button>
              )}
            </Card.Content>
          </Card>
        </List.Accordion>
        
        {/* Emergency Contacts */}
        <List.Accordion
          title="Emergency Contacts"
          left={props => <List.Icon {...props} icon="phone-alert" />}
          expanded={expandedSections.emergencyContacts}
          onPress={() => toggleSection('emergencyContacts')}
          style={styles.accordion}
        >
          <Card style={styles.sectionCard}>
            <Card.Content>
              {emergencyContacts.map((contact, index) => (
                <React.Fragment key={contact.id}>
                  <View style={styles.contactItem}>
                    <View style={styles.contactAvatar}>
                      <Avatar.Text 
                        size={40} 
                        label={contact.name.split(' ').map(n => n[0]).join('')} 
                      />
                    </View>
                    <View style={styles.contactDetails}>
                      <View style={styles.contactNameRow}>
                        <Text style={styles.contactName}>{contact.name}</Text>
                        {contact.isCaregiver && (
                          <Chip 
                            mode="outlined" 
                            style={styles.caregiverChip}
                            textStyle={{fontSize: 10}}
                          >
                            Caregiver
                          </Chip>
                        )}
                      </View>
                      <Text style={styles.contactRelationship}>{contact.relationship}</Text>
                      <Text style={styles.contactPhone}>{contact.phone}</Text>
                    </View>
                    {editMode && (
                      <IconButton
                        icon="delete"
                        size={20}
                        onPress={() => {
                          const updatedContacts = [...emergencyContacts];
                          updatedContacts.splice(index, 1);
                          setEmergencyContacts(updatedContacts);
                        }}
                        style={styles.deleteButton}
                      />
                    )}
                  </View>
                  {index < emergencyContacts.length - 1 && <Divider style={styles.divider} />}
                </React.Fragment>
              ))}
              
              {editMode && (
                <Button 
                  mode="outlined" 
                  icon="plus" 
                  onPress={() => {
                    // In a real app, this would open a form to add a new contact
                    const newContact: EmergencyContact = {
                      id: `${emergencyContacts.length + 1}`,
                      name: 'New Contact',
                      relationship: 'Relationship',
                      phone: 'Phone Number',
                      isCaregiver: false
                    };
                    setEmergencyContacts([...emergencyContacts, newContact]);
                  }}
                  style={styles.addButton}
                >
                  Add Contact
                </Button>
              )}
            </Card.Content>
          </Card>
        </List.Accordion>
        
        {/* Memory Aids (Creative Addition) */}
        <List.Accordion
          title="Memory Aids"
          left={props => <List.Icon {...props} icon="brain" />}
          expanded={expandedSections.memoryAids}
          onPress={() => toggleSection('memoryAids')}
          style={styles.accordion}
        >
          <Card style={styles.sectionCard}>
            <Card.Content>
              <Text style={styles.memoryAidsDescription}>
                Memory aids help you remember important people, places, events, and objects.
              </Text>
              
              {memoryAids.map((memoryAid, index) => (
                <React.Fragment key={memoryAid.id}>
                  <View style={styles.memoryAidItem}>
                    <View style={styles.memoryAidIconContainer}>
                      {renderMemoryAidIcon(memoryAid.type)}
                    </View>
                    <View style={styles.memoryAidDetails}>
                      <Text style={styles.memoryAidTitle}>{memoryAid.title}</Text>
                      <Text style={styles.memoryAidDescription}>{memoryAid.description}</Text>
                    </View>
                    {editMode && (
                      <IconButton
                        icon="delete"
                        size={20}
                        onPress={() => {
                          const updatedMemoryAids = [...memoryAids];
                          updatedMemoryAids.splice(index, 1);
                          setMemoryAids(updatedMemoryAids);
                        }}
                        style={styles.deleteButton}
                      />
                    )}
                  </View>
                  {index < memoryAids.length - 1 && <Divider style={styles.divider} />}
                </React.Fragment>
              ))}
              
              {editMode && (
                <Button 
                  mode="outlined" 
                  icon="plus" 
                  onPress={() => {
                    // In a real app, this would open a form to add a new memory aid
                    const newMemoryAid: MemoryAid = {
                      id: `${memoryAids.length + 1}`,
                      title: 'New Memory Aid',
                      description: 'Description',
                      date: new Date().toISOString().split('T')[0],
                      type: 'person'
                    };
                    setMemoryAids([...memoryAids, newMemoryAid]);
                  }}
                  style={styles.addButton}
                >
                  Add Memory Aid
                </Button>
              )}
            </Card.Content>
          </Card>
        </List.Accordion>
        
        {/* Account Settings */}
        <List.Accordion
          title="Account Settings"
          left={props => <List.Icon {...props} icon="cog" />}
          expanded={expandedSections.account}
          onPress={() => toggleSection('account')}
          style={styles.accordion}
        >
          <Card style={styles.sectionCard}>
            <Card.Content>
              <TouchableOpacity style={styles.accountOption}>
                <View style={styles.accountOptionIcon}>
                  <Ionicons name="lock-closed" size={24} color="#4285F4" />
                </View>
                <Text style={styles.accountOptionText}>Change Password</Text>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <Divider style={styles.divider} />
              
              <TouchableOpacity style={styles.accountOption}>
                <View style={styles.accountOptionIcon}>
                  <Ionicons name="notifications" size={24} color="#4285F4" />
                </View>
                <Text style={styles.accountOptionText}>Notification Settings</Text>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <Divider style={styles.divider} />
              
              <TouchableOpacity style={styles.accountOption}>
                <View style={styles.accountOptionIcon}>
                  <Ionicons name="cloud-download" size={24} color="#4285F4" />
                </View>
                <Text style={styles.accountOptionText}>Data Backup</Text>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <Divider style={styles.divider} />
              
              <TouchableOpacity style={styles.accountOption}>
                <View style={styles.accountOptionIcon}>
                  <Ionicons name="shield-checkmark" size={24} color="#4285F4" />
                </View>
                <Text style={styles.accountOptionText}>Privacy Settings</Text>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <Divider style={styles.divider} />
              
              <TouchableOpacity 
                style={styles.accountOption}
                onPress={handleLogout}
              >
                <View style={styles.accountOptionIcon}>
                  <Ionicons name="log-out" size={24} color="#FF3B30" />
                </View>
                <Text style={[styles.accountOptionText, { color: '#FF3B30' }]}>Log Out</Text>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <Divider style={styles.divider} />
              
              <TouchableOpacity 
                style={styles.accountOption}
                onPress={handleDeleteAccount}
              >
                <View style={styles.accountOptionIcon}>
                  <Ionicons name="trash" size={24} color="#FF3B30" />
                </View>
                <Text style={[styles.accountOptionText, { color: '#FF3B30' }]}>Delete Account</Text>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
            </Card.Content>
          </Card>
        </List.Accordion>
        
        {/* Data Sharing with Caregivers (Creative Addition) */}
        <Card style={styles.dataSharingCard}>
          <Card.Content>
            <View style={styles.dataSharingHeader}>
              <MaterialCommunityIcons name="account-group" size={24} color="#4285F4" />
              <Text style={styles.dataSharingTitle}>Caregiver Data Sharing</Text>
            </View>
            
            <Text style={styles.dataSharingDescription}>
              Control what information is shared with your caregivers
            </Text>
            
            <View style={styles.dataSharingOptions}>
              <View style={styles.dataSharingOption}>
                <Text style={styles.dataSharingOptionLabel}>Medication Tracking</Text>
                <Switch
                  value={true}
                  onValueChange={() => {}}
                  disabled={!editMode}
                  trackColor={{ false: '#D1D1D6', true: '#A2C5FF' }}
                  thumbColor={true ? '#4285F4' : '#F4F4F4'}
                />
              </View>
              
              <View style={styles.dataSharingOption}>
                <Text style={styles.dataSharingOptionLabel}>Location History</Text>
                <Switch
                  value={true}
                  onValueChange={() => {}}
                  disabled={!editMode}
                  trackColor={{ false: '#D1D1D6', true: '#A2C5FF' }}
                  thumbColor={true ? '#4285F4' : '#F4F4F4'}
                />
              </View>
              
              <View style={styles.dataSharingOption}>
                <Text style={styles.dataSharingOptionLabel}>AI Conversations</Text>
                <Switch
                  value={false}
                  onValueChange={() => {}}
                  disabled={!editMode}
                  trackColor={{ false: '#D1D1D6', true: '#A2C5FF' }}
                  thumbColor={false ? '#4285F4' : '#F4F4F4'}
                />
              </View>
              
              <View style={styles.dataSharingOption}>
                <Text style={styles.dataSharingOptionLabel}>Activity Logs</Text>
                <Switch
                  value={true}
                  onValueChange={() => {}}
                  disabled={!editMode}
                  trackColor={{ false: '#D1D1D6', true: '#A2C5FF' }}
                  thumbColor={true ? '#4285F4' : '#F4F4F4'}
                />
              </View>
            </View>
          </Card.Content>
        </Card>
        
        {/* Help & Support (Creative Addition) */}
        <Card style={styles.helpSupportCard}>
          <Card.Content>
            <View style={styles.helpSupportHeader}>
              <Ionicons name="help-buoy" size={24} color="#4285F4" />
              <Text style={styles.helpSupportTitle}>Help & Support</Text>
            </View>
            
            <View style={styles.helpSupportOptions}>
              <TouchableOpacity style={styles.helpSupportOption}>
                <Text style={styles.helpSupportOptionText}>Contact Support</Text>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.helpSupportOption}>
                <Text style={styles.helpSupportOptionText}>Video Tutorials</Text>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.helpSupportOption}>
                <Text style={styles.helpSupportOptionText}>Frequently Asked Questions</Text>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.helpSupportOption}>
                <Text style={styles.helpSupportOptionText}>Report an Issue</Text>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
            </View>
          </Card.Content>
        </Card>
        
        {/* App Information */}
        <View style={styles.appInfo}>
          <Text style={styles.appVersion}>ZAAM AI Companion v1.0.0</Text>
          <View style={styles.appInfoLinks}>
            <TouchableOpacity>
              <Text style={styles.appInfoLink}>Privacy Policy</Text>
            </TouchableOpacity>
            <Text style={styles.appInfoSeparator}>•</Text>
            <TouchableOpacity>
              <Text style={styles.appInfoLink}>Terms of Service</Text>
            </TouchableOpacity>
          </View>
        </View>
        
        {/* Bottom spacer for navigation bar */}
        <View style={styles.bottomSpacer} />
      </ScrollView>
      
      {/* Fixed Bottom Navigation */}
      <BottomNavigation />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 10,
    paddingVertical: 10,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#E1E1E1',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  scrollView: {
    flex: 1,
  },
  profileHeader: {
    alignItems: 'center',
    padding: 20,
    backgroundColor: 'white',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    marginBottom: 15,
  },
  profileImageContainer: {
    width: 100,
    height: 100,
    borderRadius: 50,
    marginBottom: 15,
    position: 'relative',
    overflow: 'hidden',
  },
  profileImage: {
    width: '100%',
    height: '100%',
    borderRadius: 50,
  },
  profileAvatar: {
    backgroundColor: '#4285F4',
  },
  editImageOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    height: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileName: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 15,
  },
  statCard: {
    width: '30%',
    elevation: 2,
    borderRadius: 10,
  },
  statCardContent: {
    alignItems: 'center',
    padding: 10,
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#4285F4',
  },
  statLabel: {
    fontSize: 12,
    color: '#8E8E93',
    textAlign: 'center',
  },
  wellnessCard: {
    width: '100%',
    marginBottom: 15,
    borderRadius: 10,
  },
  wellnessHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  wellnessTitle: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  wellnessChip: {
    backgroundColor: '#E8F1FF',
  },
  progressBarContainer: {
    height: 8,
    backgroundColor: '#E1E1E1',
    borderRadius: 4,
    marginBottom: 10,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#4285F4',
    borderRadius: 4,
  },
  wellnessDescription: {
    fontSize: 12,
    color: '#8E8E93',
  },
  accordion: {
    backgroundColor: 'white',
    marginBottom: 2,
  },
  sectionCard: {
    marginHorizontal: 15,
    marginBottom: 15,
    borderRadius: 10,
    elevation: 0,
  },
  inputRow: {
    marginBottom: 15,
  },
  inputLabel: {
    fontSize: 14,
    color: '#8E8E93',
    marginBottom: 5,
  },
  inputValue: {
    fontSize: 16,
  },
  textInput: {
    backgroundColor: 'white',
  },
  divider: {
    marginVertical: 10,
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  selectionChip: {
    marginBottom: 5,
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginVertical: 5,
  },
  switchLabel: {
    fontSize: 16,
  },
  medicationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 5,
  },
  medicationIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#E8F1FF',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  medicationDetails: {
    flex: 1,
  },
  medicationName: {
    fontSize: 16,
    fontWeight: '500',
  },
  medicationInfo: {
    fontSize: 14,
    color: '#8E8E93',
  },
  deleteButton: {
    margin: 0,
  },
  addButton: {
    marginTop: 10,
  },
  contactItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 10,
  },
  contactAvatar: {
    marginRight: 15,
  },
  contactDetails: {
    flex: 1,
  },
  contactNameRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  contactName: {
    fontSize: 16,
    fontWeight: '500',
    marginRight: 10,
  },
  caregiverChip: {
    height: 20,
  },
  contactRelationship: {
    fontSize: 14,
    color: '#8E8E93',
  },
  contactPhone: {
    fontSize: 14,
  },
  memoryAidsDescription: {
    fontSize: 14,
    color: '#8E8E93',
    marginBottom: 15,
    fontStyle: 'italic',
  },
  memoryAidItem: {
    flexDirection: 'row',
    marginVertical: 10,
  },
  memoryAidIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#E8F1FF',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  memoryAidDetails: {
    flex: 1,
  },
  memoryAidTitle: {
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 5,
  },
  memoryAidDescription: {
    fontSize: 14,
    color: '#333',
  },
  accountOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
  },
  accountOptionIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#E8F1FF',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  accountOptionText: {
    fontSize: 16,
    flex: 1,
  },
  dataSharingCard: {
    margin: 15,
    borderRadius: 10,
  },
  dataSharingHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  dataSharingTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 10,
  },
  dataSharingDescription: {
    fontSize: 14,
    color: '#8E8E93',
    marginBottom: 15,
  },
  dataSharingOptions: {
    gap: 10,
  },
  dataSharingOption: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  dataSharingOptionLabel: {
    fontSize: 16,
  },
  helpSupportCard: {
    margin: 15,
    borderRadius: 10,
  },
  helpSupportHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  helpSupportTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 10,
  },
  helpSupportOptions: {
    gap: 15,
  },
  helpSupportOption: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  helpSupportOptionText: {
    fontSize: 16,
  },
  appInfo: {
    alignItems: 'center',
    marginVertical: 20,
  },
  appVersion: {
    fontSize: 14,
    color: '#8E8E93',
    marginBottom: 5,
  },
  appInfoLinks: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  appInfoLink: {
    fontSize: 14,
    color: '#4285F4',
  },
  appInfoSeparator: {
    marginHorizontal: 5,
    color: '#8E8E93',
  },
  bottomSpacer: {
    height: 80, // Increased to account for the navigation bar height
  },
}); 