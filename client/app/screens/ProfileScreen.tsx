import React, { useState, useEffect } from 'react';
import { StyleSheet, View, ScrollView, Image, TouchableOpacity, Switch, Alert, ActivityIndicator, Modal } from 'react-native';
import { Text, Button, Card, Avatar, TextInput, Divider, List, IconButton, Chip, Portal, Dialog } from 'react-native-paper';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';

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
    type: 'person'
  });
  const [editingMemoryAidId, setEditingMemoryAidId] = useState<string | null>(null);
  
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
    medications: false,
    emergencyContacts: false,
    memoryAids: false,
    account: false
  });
  
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
              await signOut();
              console.log("User logged out successfully");
            } catch (error) {
              console.error("Logout error:", error);
              Alert.alert("Error", "Failed to log out. Please try again.");
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
      type: 'person'
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
      image_url: memoryAid.image_url
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
  
  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#4285F4" />
        <Text style={styles.loadingText}>Loading profile...</Text>
      </View>
    );
  }
  
  return (
    <View style={styles.container}>
      <ScrollView style={styles.scrollView}>
        <View style={styles.header}>
          <View style={styles.profileImageContainer}>
            {profileImage ? (
              <Image source={{ uri: profileImage }} style={styles.profileImage} />
            ) : (
              <Avatar.Text 
                size={80} 
                label={name.split(' ').map(n => n[0]).join('')} 
                style={styles.avatar}
              />
            )}
          </View>
          
          <View style={styles.profileNameContainer}>
            <Text style={styles.profileName}>{name}</Text>
            <Text style={styles.profileEmail}>{email}</Text>
          </View>
          
          <TouchableOpacity 
            style={styles.editButton}
            onPress={() => setEditMode(!editMode)}
          >
            <Text style={styles.editButtonText}>{editMode ? 'Cancel' : 'Edit'}</Text>
          </TouchableOpacity>
        </View>
        
        {/* Activity Stats */}
        <Card style={styles.statsCard}>
          <Card.Content>
            <View style={styles.statsContainer}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{activityStats.daysActive}</Text>
                <Text style={styles.statLabel}>Days Active</Text>
              </View>
              
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{activityStats.medicationAdherence}%</Text>
                <Text style={styles.statLabel}>Medication</Text>
              </View>
              
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{activityStats.aiInteractions}</Text>
                <Text style={styles.statLabel}>AI Chats</Text>
              </View>
              
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{activityStats.reminderCompletion}%</Text>
                <Text style={styles.statLabel}>Reminders</Text>
              </View>
            </View>
          </Card.Content>
        </Card>
        
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
    backgroundColor: '#F5F5F5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 15,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#E1E1E1',
  },
  profileImageContainer: {
    position: 'relative',
    marginRight: 20,
  },
  profileImage: {
    width: 80,
    height: 80,
    borderRadius: 40,
  },
  avatar: {
    backgroundColor: '#4285F4',
  },
  profileNameContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  profileName: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  profileEmail: {
    fontSize: 14,
    color: '#666',
  },
  editButton: {
    padding: 10,
    backgroundColor: '#4285F4',
    borderRadius: 5,
  },
  editButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  scrollView: {
    flex: 1,
  },
  statsCard: {
    marginHorizontal: 10,
    marginVertical: 10,
    borderRadius: 10,
    overflow: 'hidden',
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statItem: {
    alignItems: 'center',
    paddingVertical: 10,
  },
  statValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#4285F4',
    marginBottom: 5,
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
  },
  accordion: {
    marginHorizontal: 10,
    marginBottom: 10,
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: 'white',
  },
  accordionTitle: {
    fontSize: 16,
    fontWeight: '500',
  },
  accordionContent: {
    padding: 15,
  },
  input: {
    marginBottom: 10,
    backgroundColor: 'white',
  },
  saveButton: {
    marginTop: 10,
    borderRadius: 5,
    backgroundColor: '#4285F4',
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#F0F0F0',
  },
  infoLabel: {
    fontSize: 16,
    color: '#333',
    fontWeight: '500',
  },
  infoValue: {
    fontSize: 16,
    color: '#666',
    textAlign: 'right',
  },
  contactCard: {
    marginBottom: 10,
    borderRadius: 8,
    backgroundColor: 'white',
    elevation: 2,
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
    fontWeight: '500',
    marginBottom: 5,
  },
  contactDetail: {
    fontSize: 14,
    color: '#666',
    marginBottom: 2,
  },
  emptyListText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    paddingVertical: 20,
  },
  memoryCard: {
    marginBottom: 10,
    borderRadius: 8,
    backgroundColor: 'white',
    elevation: 2,
  },
  memoryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  memoryLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  memoryActions: {
    flexDirection: 'row',
  },
  memoryTitle: {
    fontSize: 16,
    fontWeight: '500',
    marginLeft: 10,
  },
  memoryDescription: {
    fontSize: 14,
    color: '#333',
    marginTop: 5,
  },
  memoryDate: {
    fontSize: 12,
    color: '#666',
    marginTop: 8,
  },
  inputLabel: {
    fontSize: 14,
    marginBottom: 8,
    marginTop: 8,
    color: '#666',
  },
  typeContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  typeButton: {
    flexDirection: 'column',
    alignItems: 'center',
    padding: 10,
    borderWidth: 1,
    borderColor: '#E1E1E1',
    borderRadius: 8,
    width: '22%',
  },
  selectedTypeButton: {
    backgroundColor: '#4285F4',
    borderColor: '#4285F4',
  },
  typeButtonText: {
    fontSize: 12,
    marginTop: 5,
    color: '#333',
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
  divider: {
    marginVertical: 10,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
    color: '#4285F4',
    marginTop: 10,
  },
  addButton: {
    marginTop: 15,
    borderColor: '#4285F4',
    borderRadius: 5,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '85%',
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 20,
    elevation: 5,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  modalInput: {
    marginBottom: 12,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  modalButton: {
    width: '48%',
  },
}); 