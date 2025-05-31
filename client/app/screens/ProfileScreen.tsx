import React, { useState, useEffect } from 'react';
import { StyleSheet, View, ScrollView, Image, TouchableOpacity, Switch, Alert, ActivityIndicator, Modal, Animated, Dimensions } from 'react-native';
import { Text, Button, Card, Avatar, TextInput, Divider, List, IconButton, Chip, Portal, Dialog } from 'react-native-paper';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import { userStatsService, UserStats } from '../services/userStatsService';
import { LinearGradient } from 'expo-linear-gradient';

const { width } = Dimensions.get('window');

interface Medication {
  id: string;
  name: string;
  dosage: string;
  frequency: string;
  time: string;
}

interface EmergencyContact {
  name: string;
  relationship: string;
  phone: string;
}

interface MemoryAid {
  _id?: string;
  title: string;
  description: string;
  date: string;
  type: 'person' | 'place' | 'event' | 'object';
  image_url?: string;
  date_of_birth?: string;  // For person type - their date of birth
  date_met_patient?: string;  // For person type - when they met the patient
  date_of_occurrence?: string;  // For event type - when the event occurred
}

export default function ProfileScreen({ setActiveTab }: { setActiveTab: (tab: string) => void }) {
  const { userData, signOut, fetchUserData, updateUserProfile, fetchMemoryAids, createMemoryAid, updateMemoryAid, deleteMemoryAid } = useAuth();
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [userProfile, setUserProfile] = useState<any>(null);
  
  // User profile state
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [profileImage, setProfileImage] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  
  // Emergency contacts state
  const [emergencyContacts, setEmergencyContacts] = useState<EmergencyContact[]>([]);
  const [contactDialogVisible, setContactDialogVisible] = useState(false);
  const [currentContact, setCurrentContact] = useState<EmergencyContact>({
    name: '',
    relationship: '',
    phone: ''
  });
  const [editingContactIndex, setEditingContactIndex] = useState<number | null>(null);
  
  // Memory aids state
  const [memoryAids, setMemoryAids] = useState<MemoryAid[]>([]);
  const [memoryAidsLoading, setMemoryAidsLoading] = useState(false);
  const [memoryAidDialogVisible, setMemoryAidDialogVisible] = useState(false);
  const [currentMemoryAid, setCurrentMemoryAid] = useState<MemoryAid>({
    title: '',
    description: '',
    date: new Date().toISOString().split('T')[0],
    type: 'person',
    date_of_birth: '',
    date_met_patient: '',
    date_of_occurrence: ''
  });
  const [editingMemoryAidId, setEditingMemoryAidId] = useState<string | null>(null);
  
  // Activity stats (now dynamic)
  const [activityStats, setActivityStats] = useState<UserStats>({
    daysActive: 0,
    aiInteractions: 0,
    medicationAdherence: 0,
    reminderCompletion: 0
  });
  const [statsLoading, setStatsLoading] = useState(true);
  
  // Sections expanded state
  const [expandedSections, setExpandedSections] = useState({
    personalInfo: true,
    medications: false,
    emergencyContacts: false,
    memoryAids: false,
    account: false
  });

  // Animation values
  const [fadeAnim] = useState(new Animated.Value(0));
  const [slideAnim] = useState(new Animated.Value(50));
  const [scaleAnim] = useState(new Animated.Value(0.9));
  
  // Initialize animations
  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 800,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 700,
        useNativeDriver: true,
      })
    ]).start();
  }, []);
  
  // Fetch user data from the backend
  useEffect(() => {
    const loadUserProfile = async () => {
      if (!userData?.id) {
        setIsLoading(false);
        return;
      }
      
      try {
        setIsLoading(true);
        const userProfileData = await fetchUserData(userData.id);
        setUserProfile(userProfileData);
        
        // Update state with fetched data
        setName(userProfileData.full_name || '');
        setAge(userProfileData.age ? String(userProfileData.age) : '');
        setGender(userProfileData.gender || '');
        
        if (userProfileData.contact_info) {
          setEmail(userProfileData.contact_info.email || '');
          setPhone(userProfileData.contact_info.phone || '');
        }
        
        if (userProfileData.emergency_contacts) {
          setEmergencyContacts(userProfileData.emergency_contacts || []);
        }
        
      } catch (error) {
        console.error('Failed to load user profile:', error);
        Alert.alert('Error', 'Failed to load user profile. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };
    
    loadUserProfile();
  }, [userData?.id]);
  
  // Fetch memory aids from the backend
  useEffect(() => {
    const loadMemoryAids = async () => {
      try {
        setMemoryAidsLoading(true);
        const data = await fetchMemoryAids();
        setMemoryAids(data);
      } catch (error) {
        console.error('Failed to load memory aids:', error);
        Alert.alert('Error', 'Failed to load memory aids. Please try again later.');
      } finally {
        setMemoryAidsLoading(false);
      }
    };
    
    if (userData?.id) {
      loadMemoryAids();
    }
  }, [userData?.id]);
  
  // Fetch user statistics
  useEffect(() => {
    const loadUserStats = async () => {
      try {
        setStatsLoading(true);
        const stats = await userStatsService.getStats();
        setActivityStats(stats);
        console.log('Loaded user stats:', stats);
      } catch (error) {
        console.error('Failed to load user stats:', error);
        // Keep default values on error
      } finally {
        setStatsLoading(false);
      }
    };

    if (userData?.id) {
      loadUserStats();
    }
  }, [userData?.id]);
  
  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections({
      ...expandedSections,
      [section]: !expandedSections[section]
    });
  };
  
  const handleSaveProfile = async () => {
    try {
      setIsSaving(true);
      
      const updatedUserData = {
        full_name: name,
        age: parseInt(age),
        gender: gender,
        contact_info: {
          email: email,
          phone: phone
        },
        emergency_contacts: emergencyContacts
      };
      
      await updateUserProfile(userData.id, updatedUserData);
      
      setEditMode(false);
      Alert.alert('Success', 'Profile updated successfully');
    } catch (error) {
      console.error('Failed to update profile:', error);
      Alert.alert('Error', 'Failed to update profile. Please try again.');
    } finally {
      setIsSaving(false);
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
          onPress: async () => {
            try {
              console.log("Starting logout process...");
              
              // Show loading state if needed
              setIsSaving(true);
              
              await signOut();
              console.log("User logged out successfully - signOut function completed");
              
              // The navigation should be handled by the AuthContext
              // but let's add a fallback just in case
              
            } catch (error) {
              console.error("Logout error:", error);
              console.error("Error details:", error instanceof Error ? error.message : String(error));
              Alert.alert("Error", "Failed to log out. Please try again.");
            } finally {
              setIsSaving(false);
            }
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
  
  // Emergency contact functions
  const openAddContactDialog = () => {
    setCurrentContact({ name: '', relationship: '', phone: '' });
    setEditingContactIndex(null);
    setContactDialogVisible(true);
  };

  const openEditContactDialog = (contact: EmergencyContact, index: number) => {
    setCurrentContact({ ...contact });
    setEditingContactIndex(index);
    setContactDialogVisible(true);
  };

  const handleDeleteContact = (index: number) => {
    Alert.alert(
      "Delete Contact",
      "Are you sure you want to delete this emergency contact?",
      [
        {
          text: "Cancel",
          style: "cancel"
        },
        {
          text: "Delete",
          onPress: async () => {
            const updatedContacts = [...emergencyContacts];
            updatedContacts.splice(index, 1);
            setEmergencyContacts(updatedContacts);
            
            // Save the updated profile with the contact removed
            try {
              setIsSaving(true);
              
              const updatedUserData = {
                full_name: name,
                age: parseInt(age) || 0,
                gender: gender,
                contact_info: {
                  email: email,
                  phone: phone
                },
                emergency_contacts: updatedContacts
              };
              
              await updateUserProfile(userData.id, updatedUserData);
              Alert.alert('Success', 'Emergency contact deleted successfully');
            } catch (error) {
              console.error('Failed to delete emergency contact:', error);
              Alert.alert('Error', 'Failed to delete emergency contact. Please try again.');
            } finally {
              setIsSaving(false);
            }
          },
          style: "destructive"
        }
      ]
    );
  };

  const handleSaveContact = async () => {
    // Validate contact data
    if (!currentContact.name || !currentContact.phone) {
      Alert.alert("Error", "Name and phone number are required.");
      return;
    }

    const updatedContacts = [...emergencyContacts];
    
    if (editingContactIndex !== null) {
      // Update existing contact
      updatedContacts[editingContactIndex] = currentContact;
    } else {
      // Add new contact
      updatedContacts.push(currentContact);
    }
    
    setEmergencyContacts(updatedContacts);
    setContactDialogVisible(false);
    
    // Save the updated profile with the new emergency contacts
    try {
      setIsSaving(true);
      
      const updatedUserData = {
        full_name: name,
        age: parseInt(age) || 0,
        gender: gender,
        contact_info: {
          email: email,
          phone: phone
        },
        emergency_contacts: updatedContacts
      };
      
      await updateUserProfile(userData.id, updatedUserData);
      Alert.alert('Success', 'Emergency contact saved successfully');
    } catch (error) {
      console.error('Failed to save emergency contact:', error);
      Alert.alert('Error', 'Failed to save emergency contact. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };
  
  // Memory Aid functions
  const openAddMemoryAidDialog = () => {
    setCurrentMemoryAid({
      title: '',
      description: '',
      date: new Date().toISOString().split('T')[0],
      type: 'person',
      date_of_birth: '',
      date_met_patient: '',
      date_of_occurrence: ''
    });
    setEditingMemoryAidId(null);
    setMemoryAidDialogVisible(true);
  };

  const openEditMemoryAidDialog = (memoryAid: MemoryAid) => {
    setCurrentMemoryAid({
      title: memoryAid.title,
      description: memoryAid.description,
      date: memoryAid.date,
      type: memoryAid.type,
      image_url: memoryAid.image_url,
      date_of_birth: memoryAid.date_of_birth || '',
      date_met_patient: memoryAid.date_met_patient || '',
      date_of_occurrence: memoryAid.date_of_occurrence || ''
    });
    setEditingMemoryAidId(memoryAid._id || null);
    setMemoryAidDialogVisible(true);
  };

  const handleDeleteMemoryAid = (memoryAidId: string) => {
    Alert.alert(
      "Delete Memory Aid",
      "Are you sure you want to delete this memory aid?",
      [
        {
          text: "Cancel",
          style: "cancel"
        },
        {
          text: "Delete",
          onPress: async () => {
            try {
              setMemoryAidsLoading(true);
              await deleteMemoryAid(memoryAidId);
              // Update the local state after successful deletion
              setMemoryAids(memoryAids.filter(aid => aid._id !== memoryAidId));
              Alert.alert('Success', 'Memory aid deleted successfully');
            } catch (error) {
              console.error('Failed to delete memory aid:', error);
              Alert.alert('Error', 'Failed to delete memory aid. Please try again.');
            } finally {
              setMemoryAidsLoading(false);
            }
          },
          style: "destructive"
        }
      ]
    );
  };

  const handleSaveMemoryAid = async () => {
    // Validate memory aid data
    if (!currentMemoryAid.title || !currentMemoryAid.type) {
      Alert.alert("Error", "Title and type are required.");
      return;
    }

    try {
      setMemoryAidsLoading(true);
      
      // Make sure date is set
      if (!currentMemoryAid.date) {
        currentMemoryAid.date = new Date().toISOString().split('T')[0];
      }
      
      if (editingMemoryAidId) {
        // Update existing memory aid
        const updatedMemoryAid = await updateMemoryAid(editingMemoryAidId, currentMemoryAid);
        
        // Update the local state
        setMemoryAids(memoryAids.map(aid => 
          aid._id === editingMemoryAidId ? updatedMemoryAid : aid
        ));
      } else {
        // Create new memory aid
        const newMemoryAid = await createMemoryAid(currentMemoryAid);
        
        // Add to the local state
        setMemoryAids([...memoryAids, newMemoryAid]);
      }
      
      setMemoryAidDialogVisible(false);
      Alert.alert('Success', editingMemoryAidId ? 'Memory aid updated successfully' : 'Memory aid added successfully');
    } catch (error) {
      console.error('Failed to save memory aid:', error);
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      Alert.alert('Error', `Failed to save memory aid: ${errorMessage}`);
    } finally {
      setMemoryAidsLoading(false);
    }
  };
  
  // Function to refresh stats (can be called manually)
  const refreshStats = async () => {
    try {
      setStatsLoading(true);
      const stats = await userStatsService.getStats();
      setActivityStats(stats);
      console.log('Refreshed user stats:', stats);
    } catch (error) {
      console.error('Failed to refresh user stats:', error);
      Alert.alert('Error', 'Failed to refresh statistics');
    } finally {
      setStatsLoading(false);
    }
  };
  
  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#6366F1" />
        <Text style={styles.loadingText}>Loading profile...</Text>
      </View>
    );
  }
  
  return (
    <View style={styles.container}>
      {/* Modern Header with Solid Blue */}
      <Animated.View 
        style={[
          styles.headerContainer, 
          {
            opacity: fadeAnim,
            transform: [{ translateY: slideAnim }]
          }
        ]}
      >
        <View style={styles.headerBackground}>
          <View style={styles.headerContent}>
            <View style={styles.profileSection}>
              <View style={styles.avatarContainer}>
                {profileImage ? (
                  <Image source={{ uri: profileImage }} style={styles.profileImage} />
                ) : (
                  <Avatar.Text 
                    size={60} 
                    label={name.split(' ').map(n => n[0]).join('')} 
                    style={styles.avatar}
                  />
                )}
                <View style={styles.onlineIndicator} />
              </View>
              
              <View style={styles.profileInfo}>
                <Text style={styles.profileName}>{name || 'User'}</Text>
                <Text style={styles.profileEmail}>{email || 'No email'}</Text>
                <View style={styles.badgeContainer}>
                  <View style={styles.badge}>
                    <Text style={styles.badgeText}>Active</Text>
                  </View>
                </View>
              </View>
            </View>
            
            <TouchableOpacity 
              style={styles.editButton}
              onPress={() => setEditMode(!editMode)}
              activeOpacity={0.8}
            >
              <Ionicons 
                name={editMode ? "close" : "pencil"} 
                size={20} 
                color="white" 
              />
            </TouchableOpacity>
          </View>
        </View>
      </Animated.View>

      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {/* Modern Activity Stats */}
        <Animated.View 
          style={[
            styles.statsSection,
            {
              opacity: fadeAnim,
              transform: [{ scale: scaleAnim }]
            }
          ]}
        >
          <View style={styles.statsHeader}>
            <Text style={styles.sectionTitle}>Activity Overview</Text>
            <TouchableOpacity 
              onPress={refreshStats} 
              disabled={statsLoading}
              style={styles.refreshButton}
            >
              <Ionicons 
                name="refresh" 
                size={20} 
                color={statsLoading ? "#CBD5E1" : "#6366F1"} 
              />
            </TouchableOpacity>
          </View>
          
          {statsLoading ? (
            <View style={styles.statsLoadingContainer}>
              <ActivityIndicator size="small" color="#6366F1" />
              <Text style={styles.statsLoadingText}>Loading statistics...</Text>
            </View>
          ) : (
            <View style={styles.statsGrid}>
              <View style={[styles.statCard, styles.statCard1]}>
                <View style={styles.statIconContainer}>
                  <Ionicons name="calendar" size={24} color="#6366F1" />
                </View>
                <Text style={styles.statValue}>{activityStats.daysActive}</Text>
                <Text style={styles.statLabel}>Days Active</Text>
              </View>
              
              <View style={[styles.statCard, styles.statCard2]}>
                <View style={styles.statIconContainer}>
                  <Ionicons name="medical" size={24} color="#10B981" />
                </View>
                <Text style={styles.statValue}>{Math.round(activityStats.medicationAdherence)}%</Text>
                <Text style={styles.statLabel}>Medication</Text>
              </View>
              
              <View style={[styles.statCard, styles.statCard3]}>
                <View style={styles.statIconContainer}>
                  <Ionicons name="chatbubbles" size={24} color="#F59E0B" />
                </View>
                <Text style={styles.statValue}>{activityStats.aiInteractions}</Text>
                <Text style={styles.statLabel}>AI Chats</Text>
              </View>
              
              <View style={[styles.statCard, styles.statCard4]}>
                <View style={styles.statIconContainer}>
                  <Ionicons name="notifications" size={24} color="#EF4444" />
                </View>
                <Text style={styles.statValue}>{Math.round(activityStats.reminderCompletion)}%</Text>
                <Text style={styles.statLabel}>Reminders</Text>
              </View>
            </View>
          )}
        </Animated.View>
        
        {/* Personal Information Section */}
        <List.Accordion
          title="Personal Information"
          expanded={expandedSections.personalInfo}
          onPress={() => toggleSection('personalInfo')}
          style={styles.accordion}
          titleStyle={styles.accordionTitle}
          left={props => <List.Icon {...props} icon="account" color="#4285F4" />}
        >
          <View style={styles.accordionContent}>
            {editMode ? (
              <>
                <TextInput
                  label="Full Name"
                  value={name}
                  onChangeText={setName}
                  mode="flat"
                  style={styles.input}
                />
                
                <TextInput
                  label="Age"
                  value={age}
                  onChangeText={setAge}
                  mode="flat"
                  keyboardType="number-pad"
                  style={styles.input}
                />
                
                <TextInput
                  label="Gender"
                  value={gender}
                  onChangeText={setGender}
                  mode="flat"
                  style={styles.input}
                />
                
                <TextInput
                  label="Email"
                  value={email}
                  onChangeText={setEmail}
                  mode="flat"
                  keyboardType="email-address"
                  style={styles.input}
                />
                
                <TextInput
                  label="Phone"
                  value={phone}
                  onChangeText={setPhone}
                  mode="flat"
                  keyboardType="phone-pad"
                  style={styles.input}
                />
                
                <Button 
                  mode="contained" 
                  onPress={handleSaveProfile}
                  style={styles.saveButton}
                  loading={isSaving}
                  disabled={isSaving}
                >
                  Save Changes
                </Button>
              </>
            ) : (
              <>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Full Name:</Text>
                  <Text style={styles.infoValue}>{name}</Text>
                </View>
                
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Age:</Text>
                  <Text style={styles.infoValue}>{age}</Text>
                </View>
                
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Gender:</Text>
                  <Text style={styles.infoValue}>{gender}</Text>
                </View>
                
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Email:</Text>
                  <Text style={styles.infoValue}>{email}</Text>
                </View>
                
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Phone:</Text>
                  <Text style={styles.infoValue}>{phone}</Text>
                </View>
              </>
            )}
          </View>
        </List.Accordion>
        
        {/* Emergency Contacts Section */}
        <List.Accordion
          title="Emergency Contacts"
          expanded={expandedSections.emergencyContacts}
          onPress={() => toggleSection('emergencyContacts')}
          style={styles.accordion}
          titleStyle={styles.accordionTitle}
          left={props => <List.Icon {...props} icon="contacts" color="#FF9500" />}
        >
          <View style={styles.accordionContent}>
            {emergencyContacts.map((contact, index) => (
              <Card key={index} style={styles.contactCard}>
                <Card.Content>
                  <View style={styles.contactHeader}>
                    <View>
                      <Text style={styles.contactName}>{contact.name}</Text>
                      <Text style={styles.contactDetail}>{contact.relationship}</Text>
                      <Text style={styles.contactDetail}>{contact.phone}</Text>
                    </View>
                    <View style={styles.contactActions}>
                      <IconButton
                        icon="pencil"
                        iconColor="#4285F4"
                        size={20}
                        onPress={() => openEditContactDialog(contact, index)}
                      />
                      <IconButton
                        icon="delete"
                        iconColor="#FF3B30"
                        size={20}
                        onPress={() => handleDeleteContact(index)}
                      />
                    </View>
                  </View>
                </Card.Content>
              </Card>
            ))}
            
            {emergencyContacts.length === 0 && (
              <Text style={styles.emptyListText}>No emergency contacts added yet.</Text>
            )}
            
            <Button 
              mode="outlined" 
              icon="plus" 
              onPress={openAddContactDialog}
              style={styles.addButton}
            >
              Add Emergency Contact
            </Button>
          </View>
        </List.Accordion>
        
        {/* Memory Aids Section */}
        <List.Accordion
          title="Memory Aids"
          expanded={expandedSections.memoryAids}
          onPress={() => toggleSection('memoryAids')}
          style={styles.accordion}
          titleStyle={styles.accordionTitle}
          left={props => <List.Icon {...props} icon="brain" color="#34A853" />}
        >
          <View style={styles.accordionContent}>
            {memoryAidsLoading ? (
              <ActivityIndicator size="small" color="#4285F4" style={styles.loader} />
            ) : (
              <>
                {memoryAids.map((aid, index) => (
                  <Card key={aid._id || index} style={styles.memoryCard}>
                    <Card.Content>
                      <View style={styles.memoryHeader}>
                        <View style={styles.memoryLeft}>
                          {renderMemoryAidIcon(aid.type)}
                          <Text style={styles.memoryTitle}>{aid.title}</Text>
                        </View>
                        <View style={styles.memoryActions}>
                          <IconButton
                            icon="pencil"
                            iconColor="#4285F4"
                            size={20}
                            onPress={() => openEditMemoryAidDialog(aid)}
                          />
                          <IconButton
                            icon="delete"
                            iconColor="#FF3B30"
                            size={20}
                            onPress={() => aid._id && handleDeleteMemoryAid(aid._id)}
                          />
                        </View>
                      </View>
                      <Text style={styles.memoryDescription}>{aid.description}</Text>
                      {aid.type === 'person' && (aid.date_of_birth || aid.date_met_patient) && (
                        <View style={styles.personDetails}>
                          {aid.date_of_birth && (
                            <Text style={styles.personDetailText}>
                              <Text style={styles.personDetailLabel}>Born: </Text>
                              {aid.date_of_birth}
                            </Text>
                          )}
                          {aid.date_met_patient && (
                            <Text style={styles.personDetailText}>
                              <Text style={styles.personDetailLabel}>Met: </Text>
                              {aid.date_met_patient}
                            </Text>
                          )}
                        </View>
                      )}
                      {aid.type === 'event' && aid.date_of_occurrence && (
                        <View style={styles.personDetails}>
                          <Text style={styles.personDetailText}>
                            <Text style={styles.personDetailLabel}>Occurred: </Text>
                            {aid.date_of_occurrence}
                          </Text>
                        </View>
                      )}
                      <Text style={styles.memoryDate}>Added: {aid.date}</Text>
                    </Card.Content>
                  </Card>
                ))}
                
                {memoryAids.length === 0 && (
                  <Text style={styles.emptyListText}>No memory aids added yet.</Text>
                )}
                
                <Button 
                  mode="outlined" 
                  icon="plus" 
                  onPress={openAddMemoryAidDialog}
                  style={styles.addButton}
                >
                  Add Memory Aid
                </Button>
              </>
            )}
          </View>
        </List.Accordion>
        
        {/* Account Section */}
        <List.Accordion
          title="Account"
          expanded={expandedSections.account}
          onPress={() => toggleSection('account')}
          style={styles.accordion}
          titleStyle={styles.accordionTitle}
          left={props => <List.Icon {...props} icon="cog" color="#8E8E93" />}
        >
          <View style={styles.accordionContent}>
            <TouchableOpacity 
              style={[styles.accountOption, isSaving && styles.disabledButton]}
              onPress={handleLogout}
              disabled={isSaving}
            >
              <View style={styles.accountOptionInfo}>
                <Ionicons name="log-out" size={24} color="#FF3B30" style={styles.accountOptionIcon} />
                <Text style={[styles.accountOptionLabel, styles.logoutText]}>
                  {isSaving ? "Logging Out..." : "Log Out"}
                </Text>
              </View>
              {isSaving ? (
                <ActivityIndicator size="small" color="#FF3B30" />
              ) : (
                <Ionicons name="chevron-forward" size={20} color="#8E8E93" />
              )}
            </TouchableOpacity>
          </View>
        </List.Accordion>
      </ScrollView>

      {/* Emergency Contact Dialog */}
      <Modal
        visible={contactDialogVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setContactDialogVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>
              {editingContactIndex !== null ? 'Edit Contact' : 'Add Contact'}
            </Text>
            
            <TextInput
              label="Name"
              value={currentContact.name}
              onChangeText={(text) => setCurrentContact({...currentContact, name: text})}
              style={styles.modalInput}
              mode="outlined"
            />
            
            <TextInput
              label="Relationship"
              value={currentContact.relationship}
              onChangeText={(text) => setCurrentContact({...currentContact, relationship: text})}
              style={styles.modalInput}
              mode="outlined"
            />
            
            <TextInput
              label="Phone Number"
              value={currentContact.phone}
              onChangeText={(text) => setCurrentContact({...currentContact, phone: text})}
              keyboardType="phone-pad"
              style={styles.modalInput}
              mode="outlined"
            />
            
            <View style={styles.modalButtons}>
              <Button 
                mode="outlined" 
                onPress={() => setContactDialogVisible(false)}
                style={styles.modalButton}
              >
                Cancel
              </Button>
              
              <Button 
                mode="contained" 
                onPress={handleSaveContact}
                style={styles.modalButton}
              >
                Save
              </Button>
            </View>
          </View>
        </View>
      </Modal>

      {/* Memory Aid Dialog */}
      <Modal
        visible={memoryAidDialogVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setMemoryAidDialogVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>
              {editingMemoryAidId ? 'Edit Memory Aid' : 'Add Memory Aid'}
            </Text>
            
            <TextInput
              label="Title"
              value={currentMemoryAid.title}
              onChangeText={(text) => setCurrentMemoryAid({...currentMemoryAid, title: text})}
              style={styles.modalInput}
              mode="outlined"
            />
            
            <TextInput
              label="Description"
              value={currentMemoryAid.description}
              onChangeText={(text) => setCurrentMemoryAid({...currentMemoryAid, description: text})}
              style={styles.modalInput}
              mode="outlined"
              multiline
              numberOfLines={3}
            />
            
            <Text style={styles.inputLabel}>Type</Text>
            <View style={styles.typeContainer}>
              <TouchableOpacity 
                style={[
                  styles.typeButton, 
                  currentMemoryAid.type === 'person' && styles.selectedTypeButton
                ]}
                onPress={() => setCurrentMemoryAid({...currentMemoryAid, type: 'person'})}
              >
                <Ionicons 
                  name="person" 
                  size={24} 
                  color={currentMemoryAid.type === 'person' ? 'white' : '#4285F4'} 
                />
                <Text style={[
                  styles.typeButtonText,
                  currentMemoryAid.type === 'person' && styles.selectedTypeButtonText
                ]}>
                  Person
                </Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[
                  styles.typeButton, 
                  currentMemoryAid.type === 'place' && styles.selectedTypeButton
                ]}
                onPress={() => setCurrentMemoryAid({...currentMemoryAid, type: 'place'})}
              >
                <Ionicons 
                  name="location" 
                  size={24} 
                  color={currentMemoryAid.type === 'place' ? 'white' : '#FF9500'} 
                />
                <Text style={[
                  styles.typeButtonText,
                  currentMemoryAid.type === 'place' && styles.selectedTypeButtonText
                ]}>
                  Place
                </Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[
                  styles.typeButton, 
                  currentMemoryAid.type === 'event' && styles.selectedTypeButton
                ]}
                onPress={() => setCurrentMemoryAid({...currentMemoryAid, type: 'event'})}
              >
                <Ionicons 
                  name="calendar" 
                  size={24} 
                  color={currentMemoryAid.type === 'event' ? 'white' : '#34C759'} 
                />
                <Text style={[
                  styles.typeButtonText,
                  currentMemoryAid.type === 'event' && styles.selectedTypeButtonText
                ]}>
                  Event
                </Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[
                  styles.typeButton, 
                  currentMemoryAid.type === 'object' && styles.selectedTypeButton
                ]}
                onPress={() => setCurrentMemoryAid({...currentMemoryAid, type: 'object'})}
              >
                <Ionicons 
                  name="cube" 
                  size={24} 
                  color={currentMemoryAid.type === 'object' ? 'white' : '#FF3B30'} 
                />
                <Text style={[
                  styles.typeButtonText,
                  currentMemoryAid.type === 'object' && styles.selectedTypeButtonText
                ]}>
                  Object
                </Text>
              </TouchableOpacity>
            </View>
            
            {/* Conditional fields for person type */}
            {currentMemoryAid.type === 'person' && (
              <>
                <TextInput
                  label="Date of Birth"
                  value={currentMemoryAid.date_of_birth || ''}
                  onChangeText={(text) => setCurrentMemoryAid({...currentMemoryAid, date_of_birth: text})}
                  style={styles.modalInput}
                  mode="outlined"
                  placeholder="YYYY-MM-DD"
                />
                
                <TextInput
                  label="When did you meet this person?"
                  value={currentMemoryAid.date_met_patient || ''}
                  onChangeText={(text) => setCurrentMemoryAid({...currentMemoryAid, date_met_patient: text})}
                  style={styles.modalInput}
                  mode="outlined"
                  placeholder="YYYY-MM-DD"
                />
              </>
            )}
            
            {/* Conditional fields for event type */}
            {currentMemoryAid.type === 'event' && (
              <TextInput
                label="When did this event occur?"
                value={currentMemoryAid.date_of_occurrence || ''}
                onChangeText={(text) => setCurrentMemoryAid({...currentMemoryAid, date_of_occurrence: text})}
                style={styles.modalInput}
                mode="outlined"
                placeholder="YYYY-MM-DD"
              />
            )}
            
            <View style={styles.modalButtons}>
              <Button 
                mode="outlined" 
                onPress={() => setMemoryAidDialogVisible(false)}
                style={styles.modalButton}
              >
                Cancel
              </Button>
              
              <Button 
                mode="contained" 
                onPress={handleSaveMemoryAid}
                style={styles.modalButton}
                loading={memoryAidsLoading}
                disabled={memoryAidsLoading}
              >
                Save
              </Button>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F8FAFC',
  },
  loadingText: {
    fontSize: 16,
    color: '#6366F1',
    marginTop: 10,
  },
  headerContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 100,
    height: 140,
  },
  headerBackground: {
    backgroundColor: '#6366F1',
    paddingTop: 50, // Account for status bar
    paddingBottom: 20,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    shadowColor: '#6366F1',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
    flex: 1,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  profileSection: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  avatarContainer: {
    position: 'relative',
    marginRight: 15,
  },
  profileImage: {
    width: 60,
    height: 60,
    borderRadius: 30,
    borderWidth: 3,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  avatar: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
  },
  onlineIndicator: {
    position: 'absolute',
    bottom: 2,
    right: 2,
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#10B981',
    borderWidth: 2,
    borderColor: 'white',
  },
  profileInfo: {
    flex: 1,
    marginRight: 10,
  },
  profileName: {
    fontSize: 18,
    fontWeight: '700',
    color: 'white',
    marginBottom: 4,
  },
  profileEmail: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    marginBottom: 8,
  },
  badgeContainer: {
    flexDirection: 'row',
  },
  badge: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  badgeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  editButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  scrollView: {
    flex: 1,
    paddingTop: 140, // Space for header
  },
  statsSection: {
    marginHorizontal: 16,
    marginTop: 20,
    marginBottom: 20,
  },
  statsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1F2937',
  },
  refreshButton: {
    padding: 8,
    borderRadius: 20,
    backgroundColor: '#F1F5F9',
  },
  statsLoadingContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 40,
  },
  statsLoadingText: {
    fontSize: 14,
    color: '#64748B',
    marginLeft: 12,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  statCard: {
    flex: 1,
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    borderWidth: 1,
    borderColor: '#F1F5F9',
  },
  statCard1: {
    borderTopWidth: 3,
    borderTopColor: '#6366F1',
  },
  statCard2: {
    borderTopWidth: 3,
    borderTopColor: '#10B981',
  },
  statCard3: {
    borderTopWidth: 3,
    borderTopColor: '#F59E0B',
  },
  statCard4: {
    borderTopWidth: 3,
    borderTopColor: '#EF4444',
  },
  statIconContainer: {
    marginBottom: 8,
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#F8FAFC',
  },
  statValue: {
    fontSize: 24,
    fontWeight: '800',
    color: '#1F2937',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#64748B',
    fontWeight: '500',
    textAlign: 'center',
  },
  accordion: {
    marginHorizontal: 16,
    marginBottom: 12,
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: 'white',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
    borderWidth: 1,
    borderColor: '#F1F5F9',
  },
  accordionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
  accordionContent: {
    padding: 20,
  },
  input: {
    marginBottom: 16,
    backgroundColor: '#F8FAFC',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  saveButton: {
    marginTop: 16,
    borderRadius: 12,
    backgroundColor: '#6366F1',
    paddingVertical: 4,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
  },
  infoLabel: {
    fontSize: 16,
    color: '#374151',
    fontWeight: '500',
  },
  infoValue: {
    fontSize: 16,
    color: '#64748B',
    textAlign: 'right',
    flex: 1,
    marginLeft: 16,
  },
  contactCard: {
    marginBottom: 12,
    borderRadius: 12,
    backgroundColor: '#F8FAFC',
    borderWidth: 1,
    borderColor: '#E2E8F0',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  contactHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  contactActions: {
    flexDirection: 'row',
  },
  contactName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 4,
  },
  contactDetail: {
    fontSize: 14,
    color: '#64748B',
    marginBottom: 2,
  },
  emptyListText: {
    fontSize: 14,
    color: '#9CA3AF',
    textAlign: 'center',
    paddingVertical: 32,
    fontStyle: 'italic',
  },
  memoryCard: {
    marginBottom: 12,
    borderRadius: 12,
    backgroundColor: 'white',
    borderWidth: 1,
    borderColor: '#E2E8F0',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  memoryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  memoryLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  memoryActions: {
    flexDirection: 'row',
  },
  memoryTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
    marginLeft: 12,
    flex: 1,
  },
  memoryDescription: {
    fontSize: 14,
    color: '#64748B',
    lineHeight: 20,
    marginTop: 4,
  },
  memoryDate: {
    fontSize: 12,
    color: '#9CA3AF',
    marginTop: 8,
    fontWeight: '500',
  },
  inputLabel: {
    fontSize: 14,
    marginBottom: 8,
    marginTop: 8,
    color: '#374151',
    fontWeight: '500',
  },
  typeContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    gap: 8,
  },
  typeButton: {
    flexDirection: 'column',
    alignItems: 'center',
    padding: 12,
    borderWidth: 1,
    borderColor: '#E2E8F0',
    borderRadius: 12,
    backgroundColor: '#F8FAFC',
    flex: 1,
  },
  selectedTypeButton: {
    backgroundColor: '#6366F1',
    borderColor: '#6366F1',
  },
  typeButtonText: {
    fontSize: 12,
    marginTop: 6,
    color: '#64748B',
    fontWeight: '500',
  },
  selectedTypeButtonText: {
    color: 'white',
  },
  loader: {
    margin: 20,
  },
  accountOption: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 16,
    paddingHorizontal: 4,
    borderRadius: 8,
  },
  accountOptionInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  accountOptionIcon: {
    marginRight: 16,
  },
  accountOptionLabel: {
    fontSize: 16,
    fontWeight: '500',
  },
  logoutText: {
    color: '#EF4444',
  },
  divider: {
    marginVertical: 10,
  },
  addButton: {
    marginTop: 16,
    borderColor: '#6366F1',
    borderRadius: 12,
    borderWidth: 2,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '90%',
    backgroundColor: 'white',
    borderRadius: 20,
    padding: 24,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 10,
    },
    shadowOpacity: 0.25,
    shadowRadius: 20,
    elevation: 10,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 20,
    textAlign: 'center',
    color: '#1F2937',
  },
  modalInput: {
    marginBottom: 16,
    backgroundColor: '#F8FAFC',
    borderRadius: 12,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20,
    gap: 12,
  },
  modalButton: {
    flex: 1,
    borderRadius: 12,
  },
  disabledButton: {
    backgroundColor: '#E5E7EB',
  },
  personDetails: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  personDetailText: {
    fontSize: 14,
    color: '#64748B',
    marginLeft: 8,
  },
  personDetailLabel: {
    fontSize: 14,
    color: '#64748B',
    fontWeight: '500',
  },
}); 