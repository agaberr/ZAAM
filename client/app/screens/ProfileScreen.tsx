import React, { useState } from 'react';
import { StyleSheet, View, ScrollView, Image, TouchableOpacity, Switch, Alert } from 'react-native';
import { Text, Button, Card, Avatar, TextInput, Divider, List, IconButton, Chip } from 'react-native-paper';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';

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

export default function ProfileScreen({ setActiveTab }) {
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
          onPress: () => console.log("Logged out"),
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
    <View style={styles.container}>
      <View style={styles.header}>
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
        {/* Profile Header */}
        <View style={styles.profileHeader}>
          <TouchableOpacity style={styles.avatarContainer}>
            {profileImage ? (
              <Avatar.Image 
                size={100} 
                source={{ uri: profileImage }} 
              />
            ) : (
              <Avatar.Text 
                size={100} 
                label={name.split(' ').map(n => n[0]).join('')} 
                backgroundColor="#4285F4" 
              />
            )}
            {editMode && (
              <View style={styles.editAvatarButton}>
                <Ionicons name="camera" size={20} color="white" />
              </View>
            )}
          </TouchableOpacity>
          
          <View style={styles.profileInfo}>
            {editMode ? (
              <TextInput
                label="Name"
                value={name}
                onChangeText={setName}
                style={styles.input}
                mode="outlined"
              />
            ) : (
              <Text style={styles.nameText}>{name}</Text>
            )}
            <Text style={styles.emailText}>{email}</Text>
            
            <View style={styles.statsContainer}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{activityStats.daysActive}</Text>
                <Text style={styles.statLabel}>Days Active</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{activityStats.medicationAdherence}%</Text>
                <Text style={styles.statLabel}>Med. Adherence</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{activityStats.aiInteractions}</Text>
                <Text style={styles.statLabel}>AI Interactions</Text>
              </View>
            </View>
          </View>
        </View>
        
        {/* Personal Information Section */}
        <Card style={styles.sectionCard}>
          <List.Accordion
            title="Personal Information"
            expanded={expandedSections.personalInfo}
            onPress={() => toggleSection('personalInfo')}
            left={props => <List.Icon {...props} icon="account" color="#4285F4" />}
            titleStyle={styles.sectionTitle}
          >
            <View style={styles.sectionContent}>
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Age:</Text>
                {editMode ? (
                  <TextInput
                    value={age}
                    onChangeText={setAge}
                    style={styles.infoInput}
                    keyboardType="number-pad"
                    mode="outlined"
                  />
                ) : (
                  <Text style={styles.infoValue}>{age}</Text>
                )}
              </View>
              
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Gender:</Text>
                {editMode ? (
                  <TextInput
                    value={gender}
                    onChangeText={setGender}
                    style={styles.infoInput}
                    mode="outlined"
                  />
                ) : (
                  <Text style={styles.infoValue}>{gender}</Text>
                )}
              </View>
              
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Phone:</Text>
                {editMode ? (
                  <TextInput
                    value={phone}
                    onChangeText={setPhone}
                    style={styles.infoInput}
                    keyboardType="phone-pad"
                    mode="outlined"
                  />
                ) : (
                  <Text style={styles.infoValue}>{phone}</Text>
                )}
              </View>
              
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Email:</Text>
                {editMode ? (
                  <TextInput
                    value={email}
                    onChangeText={setEmail}
                    style={styles.infoInput}
                    keyboardType="email-address"
                    mode="outlined"
                  />
                ) : (
                  <Text style={styles.infoValue}>{email}</Text>
                )}
              </View>
            </View>
          </List.Accordion>
        </Card>
        
        {/* Medications Section */}
        <Card style={styles.sectionCard}>
          <List.Accordion
            title="Medications"
            expanded={expandedSections.medications}
            onPress={() => toggleSection('medications')}
            left={props => <List.Icon {...props} icon="pill" color="#FF9500" />}
            titleStyle={styles.sectionTitle}
          >
            <View style={styles.sectionContent}>
              {medications.map((medication, index) => (
                <View key={medication.id} style={styles.medicationItem}>
                  <View style={styles.medicationHeader}>
                    <View style={styles.medicationTitleContainer}>
                      <Text style={styles.medicationName}>{medication.name}</Text>
                      <Text style={styles.medicationDosage}>{medication.dosage}</Text>
                    </View>
                    {editMode && (
                      <IconButton
                        icon="pencil"
                        size={20}
                        color="#4285F4"
                        onPress={() => console.log('Edit medication')}
                      />
                    )}
                  </View>
                  <View style={styles.medicationDetails}>
                    <View style={styles.medicationDetail}>
                      <Ionicons name="time-outline" size={16} color="#777" />
                      <Text style={styles.medicationDetailText}>{medication.time}</Text>
                    </View>
                    <View style={styles.medicationDetail}>
                      <Ionicons name="repeat" size={16} color="#777" />
                      <Text style={styles.medicationDetailText}>{medication.frequency}</Text>
                    </View>
                  </View>
                  {index < medications.length - 1 && <Divider style={styles.divider} />}
                </View>
              ))}
              
              {editMode && (
                <Button 
                  mode="outlined" 
                  icon="plus" 
                  onPress={() => console.log('Add medication')}
                  style={styles.addButton}
                >
                  Add Medication
                </Button>
              )}
            </View>
          </List.Accordion>
        </Card>
        
        {/* Emergency Contacts Section */}
        <Card style={styles.sectionCard}>
          <List.Accordion
            title="Emergency Contacts"
            expanded={expandedSections.emergencyContacts}
            onPress={() => toggleSection('emergencyContacts')}
            left={props => <List.Icon {...props} icon="phone-alert" color="#FF3B30" />}
            titleStyle={styles.sectionTitle}
          >
            <View style={styles.sectionContent}>
              {emergencyContacts.map((contact, index) => (
                <View key={contact.id} style={styles.contactItem}>
                  <View style={styles.contactHeader}>
                    <View style={styles.contactInfo}>
                      <Text style={styles.contactName}>{contact.name}</Text>
                      <View style={styles.contactRelationshipContainer}>
                        <Text style={styles.contactRelationship}>{contact.relationship}</Text>
                        {contact.isCaregiver && (
                          <Chip style={styles.caregiverChip} textStyle={styles.caregiverChipText}>
                            Caregiver
                          </Chip>
                        )}
                      </View>
                    </View>
                    <View style={styles.contactActions}>
                      <IconButton
                        icon="phone"
                        size={20}
                        color="#4285F4"
                        onPress={() => console.log('Call contact')}
                      />
                      {editMode && (
                        <IconButton
                          icon="pencil"
                          size={20}
                          color="#4285F4"
                          onPress={() => console.log('Edit contact')}
                        />
                      )}
                    </View>
                  </View>
                  <Text style={styles.contactPhone}>{contact.phone}</Text>
                  {index < emergencyContacts.length - 1 && <Divider style={styles.divider} />}
                </View>
              ))}
              
              {editMode && (
                <Button 
                  mode="outlined" 
                  icon="plus" 
                  onPress={() => console.log('Add contact')}
                  style={styles.addButton}
                >
                  Add Contact
                </Button>
              )}
            </View>
          </List.Accordion>
        </Card>
        
        {/* Memory Aids Section */}
        <Card style={styles.sectionCard}>
          <List.Accordion
            title="Memory Aids"
            expanded={expandedSections.memoryAids}
            onPress={() => toggleSection('memoryAids')}
            left={props => <List.Icon {...props} icon="brain" color="#34C759" />}
            titleStyle={styles.sectionTitle}
          >
            <View style={styles.sectionContent}>
              {memoryAids.map((memoryAid, index) => (
                <View key={memoryAid.id} style={styles.memoryAidItem}>
                  <View style={styles.memoryAidHeader}>
                    <View style={styles.memoryAidIconContainer}>
                      {renderMemoryAidIcon(memoryAid.type)}
                    </View>
                    <View style={styles.memoryAidInfo}>
                      <Text style={styles.memoryAidTitle}>{memoryAid.title}</Text>
                      <Text style={styles.memoryAidDate}>Added on {new Date(memoryAid.date).toLocaleDateString()}</Text>
                    </View>
                    {editMode && (
                      <IconButton
                        icon="pencil"
                        size={20}
                        color="#4285F4"
                        onPress={() => console.log('Edit memory aid')}
                      />
                    )}
                  </View>
                  <Text style={styles.memoryAidDescription}>{memoryAid.description}</Text>
                  {index < memoryAids.length - 1 && <Divider style={styles.divider} />}
                </View>
              ))}
              
              {editMode && (
                <Button 
                  mode="outlined" 
                  icon="plus" 
                  onPress={() => console.log('Add memory aid')}
                  style={styles.addButton}
                >
                  Add Memory Aid
                </Button>
              )}
            </View>
          </List.Accordion>
        </Card>
        
        {/* Preferences Section */}
        <Card style={styles.sectionCard}>
          <List.Accordion
            title="Preferences"
            expanded={expandedSections.preferences}
            onPress={() => toggleSection('preferences')}
            left={props => <List.Icon {...props} icon="cog" color="#8E8E93" />}
            titleStyle={styles.sectionTitle}
          >
            <View style={styles.sectionContent}>
              <View style={styles.preferenceItem}>
                <Text style={styles.preferenceLabel}>Text Size</Text>
                <View style={styles.textSizeOptions}>
                  <TouchableOpacity 
                    style={[
                      styles.textSizeOption, 
                      textSize === 'Small' && styles.selectedTextSize
                    ]}
                    onPress={() => setTextSize('Small')}
                  >
                    <Text style={[
                      styles.textSizeOptionText,
                      { fontSize: 12 },
                      textSize === 'Small' && styles.selectedTextSizeText
                    ]}>
                      Small
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity 
                    style={[
                      styles.textSizeOption, 
                      textSize === 'Medium' && styles.selectedTextSize
                    ]}
                    onPress={() => setTextSize('Medium')}
                  >
                    <Text style={[
                      styles.textSizeOptionText,
                      { fontSize: 14 },
                      textSize === 'Medium' && styles.selectedTextSizeText
                    ]}>
                      Medium
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity 
                    style={[
                      styles.textSizeOption, 
                      textSize === 'Large' && styles.selectedTextSize
                    ]}
                    onPress={() => setTextSize('Large')}
                  >
                    <Text style={[
                      styles.textSizeOptionText,
                      { fontSize: 16 },
                      textSize === 'Large' && styles.selectedTextSizeText
                    ]}>
                      Large
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.preferenceItem}>
                <Text style={styles.preferenceLabel}>Voice Type</Text>
                <View style={styles.voiceTypeOptions}>
                  <TouchableOpacity 
                    style={[
                      styles.voiceTypeOption, 
                      voiceType === 'Male' && styles.selectedVoiceType
                    ]}
                    onPress={() => setVoiceType('Male')}
                  >
                    <Ionicons 
                      name="man" 
                      size={24} 
                      color={voiceType === 'Male' ? 'white' : '#4285F4'} 
                    />
                    <Text style={[
                      styles.voiceTypeText,
                      voiceType === 'Male' && styles.selectedVoiceTypeText
                    ]}>
                      Male
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity 
                    style={[
                      styles.voiceTypeOption, 
                      voiceType === 'Female' && styles.selectedVoiceType
                    ]}
                    onPress={() => setVoiceType('Female')}
                  >
                    <Ionicons 
                      name="woman" 
                      size={24} 
                      color={voiceType === 'Female' ? 'white' : '#4285F4'} 
                    />
                    <Text style={[
                      styles.voiceTypeText,
                      voiceType === 'Female' && styles.selectedVoiceTypeText
                    ]}>
                      Female
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.switchPreferenceItem}>
                <View style={styles.switchPreferenceInfo}>
                  <Ionicons name="contrast" size={24} color="#4285F4" style={styles.switchPreferenceIcon} />
                  <Text style={styles.switchPreferenceLabel}>High Contrast Mode</Text>
                </View>
                <Switch
                  value={highContrastMode}
                  onValueChange={setHighContrastMode}
                  trackColor={{ false: '#D1D1D6', true: '#4285F4' }}
                  thumbColor="white"
                />
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.switchPreferenceItem}>
                <View style={styles.switchPreferenceInfo}>
                  <Ionicons name="mic" size={24} color="#4285F4" style={styles.switchPreferenceIcon} />
                  <Text style={styles.switchPreferenceLabel}>Voice Commands</Text>
                </View>
                <Switch
                  value={voiceCommandsEnabled}
                  onValueChange={setVoiceCommandsEnabled}
                  trackColor={{ false: '#D1D1D6', true: '#4285F4' }}
                  thumbColor="white"
                />
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.switchPreferenceItem}>
                <View style={styles.switchPreferenceInfo}>
                  <Ionicons name="location" size={24} color="#4285F4" style={styles.switchPreferenceIcon} />
                  <Text style={styles.switchPreferenceLabel}>Location Tracking</Text>
                </View>
                <Switch
                  value={locationTrackingEnabled}
                  onValueChange={setLocationTrackingEnabled}
                  trackColor={{ false: '#D1D1D6', true: '#4285F4' }}
                  thumbColor="white"
                />
              </View>
              
              <Divider style={styles.divider} />
              
              <View style={styles.switchPreferenceItem}>
                <View style={styles.switchPreferenceInfo}>
                  <Ionicons name="moon" size={24} color="#4285F4" style={styles.switchPreferenceIcon} />
                  <Text style={styles.switchPreferenceLabel}>Night Mode</Text>
                </View>
                <Switch
                  value={nightModeEnabled}
                  onValueChange={setNightModeEnabled}
                  trackColor={{ false: '#D1D1D6', true: '#4285F4' }}
                  thumbColor="white"
                />
              </View>
            </View>
          </List.Accordion>
        </Card>
        
        {/* Account Section */}
        <Card style={styles.sectionCard}>
          <List.Accordion
            title="Account"
            expanded={expandedSections.account}
            onPress={() => toggleSection('account')}
            left={props => <List.Icon {...props} icon="shield-account" color="#8E8E93" />}
            titleStyle={styles.sectionTitle}
          >
            <View style={styles.sectionContent}>
              <TouchableOpacity style={styles.accountOption}>
                <View style={styles.accountOptionInfo}>
                  <Ionicons name="lock-closed" size={24} color="#4285F4" style={styles.accountOptionIcon} />
                  <Text style={styles.accountOptionLabel}>Change Password</Text>
                </View>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <Divider style={styles.divider} />
              
              <TouchableOpacity style={styles.accountOption}>
                <View style={styles.accountOptionInfo}>
                  <Ionicons name="cloud-download" size={24} color="#4285F4" style={styles.accountOptionIcon} />
                  <Text style={styles.accountOptionLabel}>Export Data</Text>
                </View>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <Divider style={styles.divider} />
              
              <TouchableOpacity style={styles.accountOption}>
                <View style={styles.accountOptionInfo}>
                  <Ionicons name="help-circle" size={24} color="#4285F4" style={styles.accountOptionIcon} />
                  <Text style={styles.accountOptionLabel}>Help & Support</Text>
                </View>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
              
              <Divider style={styles.divider} />
              
              <TouchableOpacity 
                style={styles.accountOption}
                onPress={handleLogout}
              >
                <View style={styles.accountOptionInfo}>
                  <Ionicons name="log-out" size={24} color="#FF3B30" style={styles.accountOptionIcon} />
                  <Text style={[styles.accountOptionLabel, styles.logoutText]}>Log Out</Text>
                </View>
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              </TouchableOpacity>
            </View>
          </List.Accordion>
        </Card>
        
        <View style={styles.footer}>
          <Text style={styles.versionText}>Version 1.0.0</Text>
          <Text style={styles.copyrightText}>© 2023 MemoryCare</Text>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 10,
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
    flexDirection: 'row',
    padding: 20,
    backgroundColor: 'white',
    marginBottom: 10,
  },
  avatarContainer: {
    position: 'relative',
    marginRight: 20,
  },
  editAvatarButton: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    backgroundColor: '#4285F4',
    borderRadius: 15,
    width: 30,
    height: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileInfo: {
    flex: 1,
    justifyContent: 'center',
  },
  nameText: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  emailText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 15,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#4285F4',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
  },
  sectionCard: {
    marginHorizontal: 10,
    marginBottom: 10,
    borderRadius: 10,
    overflow: 'hidden',
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '500',
  },
  sectionContent: {
    padding: 15,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  infoLabel: {
    fontSize: 16,
    color: '#666',
    width: '30%',
  },
  infoValue: {
    fontSize: 16,
    flex: 1,
    textAlign: 'right',
  },
  infoInput: {
    flex: 1,
    height: 40,
    fontSize: 16,
  },
  input: {
    marginBottom: 10,
  },
  divider: {
    marginVertical: 10,
  },
  medicationItem: {
    marginBottom: 15,
  },
  medicationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  medicationTitleContainer: {
    flex: 1,
  },
  medicationName: {
    fontSize: 16,
    fontWeight: '500',
  },
  medicationDosage: {
    fontSize: 14,
    color: '#666',
  },
  medicationDetails: {
    flexDirection: 'row',
    marginTop: 5,
  },
  medicationDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 15,
  },
  medicationDetailText: {
    fontSize: 14,
    color: '#666',
    marginLeft: 5,
  },
  contactItem: {
    marginBottom: 15,
  },
  contactHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  contactInfo: {
    flex: 1,
  },
  contactName: {
    fontSize: 16,
    fontWeight: '500',
  },
  contactRelationshipContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 2,
  },
  contactRelationship: {
    fontSize: 14,
    color: '#666',
    marginRight: 10,
  },
  contactPhone: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  contactActions: {
    flexDirection: 'row',
  },
  caregiverChip: {
    backgroundColor: '#E8F1FF',
    height: 24,
  },
  caregiverChipText: {
    fontSize: 12,
    color: '#4285F4',
  },
  memoryAidItem: {
    marginBottom: 15,
  },
  memoryAidHeader: {
    flexDirection: 'row',
    alignItems: 'center',
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
  memoryAidInfo: {
    flex: 1,
  },
  memoryAidTitle: {
    fontSize: 16,
    fontWeight: '500',
  },
  memoryAidDate: {
    fontSize: 12,
    color: '#666',
  },
  memoryAidDescription: {
    fontSize: 14,
    color: '#333',
    marginTop: 10,
    marginLeft: 55,
  },
  preferenceItem: {
    marginBottom: 15,
  },
  preferenceLabel: {
    fontSize: 16,
    marginBottom: 10,
  },
  textSizeOptions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  textSizeOption: {
    flex: 1,
    padding: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E1E1E1',
    borderRadius: 5,
    marginHorizontal: 5,
  },
  selectedTextSize: {
    borderColor: '#4285F4',
    backgroundColor: '#E8F1FF',
  },
  textSizeOptionText: {
    color: '#333',
  },
  selectedTextSizeText: {
    color: '#4285F4',
    fontWeight: '500',
  },
  voiceTypeOptions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  voiceTypeOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    borderWidth: 1,
    borderColor: '#E1E1E1',
    borderRadius: 20,
    width: '45%',
    justifyContent: 'center',
  },
  selectedVoiceType: {
    borderColor: '#4285F4',
    backgroundColor: '#4285F4',
  },
  voiceTypeText: {
    marginLeft: 10,
    fontSize: 16,
    color: '#4285F4',
  },
  selectedVoiceTypeText: {
    color: 'white',
  },
  switchPreferenceItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginVertical: 10,
  },
  switchPreferenceInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  switchPreferenceIcon: {
    marginRight: 15,
  },
  switchPreferenceLabel: {
    fontSize: 16,
  },
  accountOption: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
  },
  accountOptionInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  accountOptionIcon: {
    marginRight: 15,
  },
  accountOptionLabel: {
    fontSize: 16,
  },
  logoutText: {
    color: '#FF3B30',
  },
  addButton: {
    marginTop: 10,
    borderColor: '#4285F4',
  },
  footer: {
    padding: 20,
    alignItems: 'center',
  },
  versionText: {
    fontSize: 14,
    color: '#8E8E93',
    marginBottom: 5,
  },
  copyrightText: {
    fontSize: 12,
    color: '#8E8E93',
  },
}); 