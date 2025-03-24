import React, { useState, useEffect } from 'react';
import { StyleSheet, View, ScrollView, TouchableOpacity, Alert } from 'react-native';
import { TextInput, Button, Text, IconButton, Chip, Divider, List, Dialog, Portal } from 'react-native-paper';
import { router } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../context/AuthContext';
import { Ionicons } from '@expo/vector-icons';

type EmergencyContact = {
  name: string;
  relationship: string;
  phone: string;
};

export default function UserDataScreen() {
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [phone, setPhone] = useState('');
  const [loading, setLoading] = useState(false);
  const [emergencyContacts, setEmergencyContacts] = useState<EmergencyContact[]>([]);
  
  // Emergency contact dialog
  const [contactDialogVisible, setContactDialogVisible] = useState(false);
  const [contactName, setContactName] = useState('');
  const [contactRelationship, setContactRelationship] = useState('');
  const [contactPhone, setContactPhone] = useState('');
  const [editingContactIndex, setEditingContactIndex] = useState<number | null>(null);
  
  const { completeOnboarding, tempRegData } = useAuth();

  // Check if we have temporary registration data
  useEffect(() => {
    if (!tempRegData) {
      Alert.alert(
        "Error",
        "No registration data found. Please sign up first.",
        [{ text: "OK", onPress: () => router.replace('/auth/signup') }]
      );
    }
  }, [tempRegData]);

  const handleComplete = async () => {
    // Validate required fields
    if (!age) {
      Alert.alert("Error", "Please enter your age");
      return;
    }
    if (!gender) {
      Alert.alert("Error", "Please select your gender");
      return;
    }
    if (!phone) {
      Alert.alert("Error", "Please enter your phone number");
      return;
    }
    
    setLoading(true);
    
    try {
      // Convert age to number
      const ageNumber = parseInt(age, 10);
      
      // Submit data to backend via context
      await completeOnboarding({
        age: ageNumber,
        gender,
        phone,
        emergency_contacts: emergencyContacts
      });
      
      // No need to navigate here - the auth context will handle it
    } catch (error) {
      console.error('Failed to complete registration:', error);
      Alert.alert("Error", error instanceof Error ? error.message : "Failed to complete registration");
    } finally {
      setLoading(false);
    }
  };

  const goBack = () => {
    router.back();
  };
  
  // Emergency contact handlers
  const showAddContactDialog = () => {
    setContactName('');
    setContactRelationship('');
    setContactPhone('');
    setEditingContactIndex(null);
    setContactDialogVisible(true);
  };
  
  const showEditContactDialog = (index: number) => {
    const contact = emergencyContacts[index];
    setContactName(contact.name);
    setContactRelationship(contact.relationship);
    setContactPhone(contact.phone);
    setEditingContactIndex(index);
    setContactDialogVisible(true);
  };
  
  const hideContactDialog = () => {
    setContactDialogVisible(false);
  };
  
  const addOrUpdateContact = () => {
    if (!contactName || !contactRelationship || !contactPhone) {
      Alert.alert("Error", "Please fill in all contact fields");
      return;
    }
    
    const newContact = {
      name: contactName,
      relationship: contactRelationship,
      phone: contactPhone
    };
    
    if (editingContactIndex !== null) {
      // Update existing contact
      const updatedContacts = [...emergencyContacts];
      updatedContacts[editingContactIndex] = newContact;
      setEmergencyContacts(updatedContacts);
    } else {
      // Add new contact
      setEmergencyContacts([...emergencyContacts, newContact]);
    }
    
    hideContactDialog();
  };
  
  const deleteContact = (index: number) => {
    Alert.alert(
      "Delete Contact",
      "Are you sure you want to delete this emergency contact?",
      [
        { text: "Cancel", style: "cancel" },
        { 
          text: "Delete", 
          style: "destructive",
          onPress: () => {
            const updatedContacts = [...emergencyContacts];
            updatedContacts.splice(index, 1);
            setEmergencyContacts(updatedContacts);
          }
        }
      ]
    );
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#4285F4', '#34A853']}
        style={styles.background}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      />
      
      <IconButton
        icon="arrow-left"
        size={24}
        onPress={goBack}
        style={styles.backButton}
        iconColor="white"
      />
      
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.content}>
          <Text variant="headlineMedium" style={styles.title}>Complete Your Profile</Text>
          <Text variant="bodyLarge" style={styles.subtitle}>
            Tell us more about yourself
          </Text>
          
          <View style={styles.form}>
            <TextInput
              label="Age"
              value={age}
              onChangeText={setAge}
              mode="flat"
              style={styles.input}
              keyboardType="number-pad"
              underlineColor="rgba(255,255,255,0.3)"
              activeUnderlineColor="white"
              textColor="white"
              theme={{ 
                colors: { 
                  onSurfaceVariant: 'white',
                  text: 'white',
                  placeholder: 'white' 
                } 
              }}
            />
            
            <Text style={styles.sectionTitle}>Gender</Text>
            <View style={styles.chipContainer}>
              <Chip
                selected={gender === 'Male'}
                onPress={() => setGender('Male')}
                style={[
                  styles.chip,
                  gender === 'Male' && styles.selectedChip
                ]}
                textStyle={[
                  styles.chipText,
                  gender === 'Male' && styles.selectedChipText
                ]}
              >
                Male
              </Chip>
              
              <Chip
                selected={gender === 'Female'}
                onPress={() => setGender('Female')}
                style={[
                  styles.chip,
                  gender === 'Female' && styles.selectedChip
                ]}
                textStyle={[
                  styles.chipText,
                  gender === 'Female' && styles.selectedChipText
                ]}
              >
                Female
              </Chip>
              
              <Chip
                selected={gender === 'Other'}
                onPress={() => setGender('Other')}
                style={[
                  styles.chip,
                  gender === 'Other' && styles.selectedChip
                ]}
                textStyle={[
                  styles.chipText,
                  gender === 'Other' && styles.selectedChipText
                ]}
              >
                Other
              </Chip>
            </View>
            
            <TextInput
              label="Phone Number"
              value={phone}
              onChangeText={setPhone}
              mode="flat"
              style={styles.input}
              keyboardType="phone-pad"
              underlineColor="rgba(255,255,255,0.3)"
              activeUnderlineColor="white"
              textColor="white"
              theme={{ 
                colors: { 
                  onSurfaceVariant: 'white',
                  text: 'white',
                  placeholder: 'white' 
                } 
              }}
            />
            
            <View style={styles.emergencyContactsHeader}>
              <Text style={styles.sectionTitle}>Emergency Contacts</Text>
              <IconButton
                icon="plus"
                size={20}
                iconColor="white"
                style={styles.addButton}
                onPress={showAddContactDialog}
              />
            </View>
            
            {emergencyContacts.length === 0 ? (
              <Text style={styles.noContactsText}>No emergency contacts added yet.</Text>
            ) : (
              emergencyContacts.map((contact, index) => (
                <View key={index} style={styles.contactCard}>
                  <View style={styles.contactInfo}>
                    <Text style={styles.contactName}>{contact.name}</Text>
                    <Text style={styles.contactDetail}>{contact.relationship}</Text>
                    <Text style={styles.contactDetail}>{contact.phone}</Text>
                  </View>
                  <View style={styles.contactActions}>
                    <IconButton
                      icon="pencil"
                      size={18}
                      iconColor="white"
                      onPress={() => showEditContactDialog(index)}
                    />
                    <IconButton
                      icon="delete"
                      size={18}
                      iconColor="white"
                      onPress={() => deleteContact(index)}
                    />
                  </View>
                </View>
              ))
            )}
            
            <Button
              mode="contained"
              onPress={handleComplete}
              style={styles.button}
              contentStyle={styles.buttonContent}
              labelStyle={styles.buttonLabel}
              loading={loading}
              disabled={loading}
            >
              Complete Registration
            </Button>
          </View>
        </View>
      </ScrollView>
      
      {/* Emergency Contact Dialog */}
      <Portal>
        <Dialog visible={contactDialogVisible} onDismiss={hideContactDialog} style={styles.dialog}>
          <Dialog.Title>{editingContactIndex !== null ? 'Edit Contact' : 'Add Emergency Contact'}</Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Name"
              value={contactName}
              onChangeText={setContactName}
              style={styles.dialogInput}
            />
            <TextInput
              label="Relationship"
              value={contactRelationship}
              onChangeText={setContactRelationship}
              style={styles.dialogInput}
              placeholder="e.g. Spouse, Child, Friend"
            />
            <TextInput
              label="Phone Number"
              value={contactPhone}
              onChangeText={setContactPhone}
              style={styles.dialogInput}
              keyboardType="phone-pad"
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={hideContactDialog}>Cancel</Button>
            <Button onPress={addOrUpdateContact}>Save</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  background: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  backButton: {
    position: 'absolute',
    top: 40,
    left: 16,
    zIndex: 10,
  },
  scrollContent: {
    flexGrow: 1,
    paddingTop: 80,
    paddingBottom: 40,
  },
  content: {
    padding: 24,
    width: '100%',
    maxWidth: 400,
    alignSelf: 'center',
  },
  title: {
    color: 'white',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    color: 'rgba(255,255,255,0.8)',
    marginBottom: 32,
  },
  form: {
    width: '100%',
  },
  input: {
    marginBottom: 24,
    backgroundColor: 'transparent',
    height: 60,
    paddingBottom: 8,
  },
  sectionTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 12,
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 24,
  },
  chip: {
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  selectedChip: {
    backgroundColor: 'white',
  },
  chipText: {
    color: 'white',
  },
  selectedChipText: {
    color: '#4285F4',
  },
  emergencyContactsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  addButton: {
    margin: 0,
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  noContactsText: {
    color: 'rgba(255,255,255,0.6)',
    fontStyle: 'italic',
    marginBottom: 24,
    textAlign: 'center',
  },
  contactCard: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  contactInfo: {
    flex: 1,
  },
  contactName: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 4,
  },
  contactDetail: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 14,
  },
  contactActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dialog: {
    backgroundColor: 'white',
  },
  dialogInput: {
    marginBottom: 16,
  },
  button: {
    marginTop: 16,
    backgroundColor: 'white',
    borderRadius: 12,
  },
  buttonContent: {
    height: 56,
  },
  buttonLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#4285F4',
  },
}); 