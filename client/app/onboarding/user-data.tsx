import React, { useState, useEffect } from 'react';
import { StyleSheet, View, ScrollView, TouchableOpacity, Alert, KeyboardAvoidingView, Platform, Dimensions } from 'react-native';
import { TextInput, Button, Text, IconButton, Chip, Dialog, Portal } from 'react-native-paper';
import { router } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../context/AuthContext';
import { Ionicons } from '@expo/vector-icons';

const { width, height } = Dimensions.get('window');

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
      {/* Main gradient background */}
      <LinearGradient
        colors={['#1e3c72', '#2a5298', '#4facfe']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.backgroundGradient}
      />
      
      {/* Decorative geometric shapes */}
      <View style={styles.decorativeShapes}>
        <View style={[styles.circle, styles.circle1]} />
        <View style={[styles.circle, styles.circle2]} />
        <View style={[styles.circle, styles.circle3]} />
        <LinearGradient
          colors={['rgba(255, 255, 255, 0.1)', 'rgba(255, 255, 255, 0.05)']}
          style={[styles.shape, styles.shape1]}
        />
        <LinearGradient
          colors={['rgba(255, 255, 255, 0.08)', 'rgba(255, 255, 255, 0.03)']}
          style={[styles.shape, styles.shape2]}
        />
      </View>

      {/* Content overlay */}
      <LinearGradient
        colors={['rgba(0, 0, 0, 0.1)', 'rgba(0, 0, 0, 0.05)', 'rgba(0, 0, 0, 0.1)']}
        style={styles.contentOverlay}
      />
      
      <IconButton
        icon="arrow-left"
        size={24}
        onPress={goBack}
        style={styles.backButton}
        iconColor="white"
      />
      
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardAvoidingView}
      >
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <View style={styles.content}>
            <Text variant="headlineMedium" style={styles.title}>Complete Your Profile</Text>
            <Text variant="bodyLarge" style={styles.subtitle}>
              Tell us more about yourself to personalize your ZAAM experience
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
                <TouchableOpacity
                  style={styles.addButton}
                  onPress={showAddContactDialog}
                >
                  <Ionicons name="add" size={20} color="white" />
                </TouchableOpacity>
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
                      <TouchableOpacity
                        onPress={() => showEditContactDialog(index)}
                        style={styles.actionButton}
                      >
                        <Ionicons name="pencil" size={16} color="white" />
                      </TouchableOpacity>
                      <TouchableOpacity
                        onPress={() => deleteContact(index)}
                        style={styles.actionButton}
                      >
                        <Ionicons name="trash" size={16} color="white" />
                      </TouchableOpacity>
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
      </KeyboardAvoidingView>
      
      {/* Emergency Contact Dialog */}
      <Portal>
        <Dialog visible={contactDialogVisible} onDismiss={hideContactDialog} style={styles.dialog}>
          <Dialog.Title style={styles.dialogTitle}>
            {editingContactIndex !== null ? 'Edit Contact' : 'Add Emergency Contact'}
          </Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Name"
              value={contactName}
              onChangeText={setContactName}
              style={styles.dialogInput}
              mode="outlined"
            />
            <TextInput
              label="Relationship"
              value={contactRelationship}
              onChangeText={setContactRelationship}
              style={styles.dialogInput}
              placeholder="e.g. Spouse, Child, Friend"
              mode="outlined"
            />
            <TextInput
              label="Phone Number"
              value={contactPhone}
              onChangeText={setContactPhone}
              style={styles.dialogInput}
              keyboardType="phone-pad"
              mode="outlined"
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={hideContactDialog} textColor="#666">Cancel</Button>
            <Button onPress={addOrUpdateContact} textColor="#1e3c72">Save</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1e3c72',
  },
  backgroundGradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  decorativeShapes: {
    position: 'absolute',
    width: '100%',
    height: '100%',
  },
  circle: {
    position: 'absolute',
    borderRadius: 1000,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  circle1: {
    width: 200,
    height: 200,
    top: -100,
    right: -50,
  },
  circle2: {
    width: 150,
    height: 150,
    bottom: 100,
    left: -75,
  },
  circle3: {
    width: 100,
    height: 100,
    top: height * 0.3,
    right: 20,
  },
  shape: {
    position: 'absolute',
  },
  shape1: {
    width: 300,
    height: 300,
    borderRadius: 150,
    top: -150,
    left: -100,
    transform: [{ rotate: '45deg' }],
  },
  shape2: {
    width: 250,
    height: 250,
    borderRadius: 125,
    bottom: -125,
    right: -50,
    transform: [{ rotate: '-30deg' }],
  },
  contentOverlay: {
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
  keyboardAvoidingView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingTop: 80,
    paddingBottom: 40,
  },
  content: {
    padding: 24,
    width: '100%',
    maxWidth: 400,
    alignSelf: 'center',
    zIndex: 5,
  },
  title: {
    color: 'white',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    color: 'rgba(255,255,255,0.9)',
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
    fontWeight: '600',
    marginBottom: 12,
    marginTop: 8,
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 24,
  },
  chip: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 20,
  },
  selectedChip: {
    backgroundColor: 'white',
  },
  chipText: {
    color: 'white',
    fontWeight: '500',
  },
  selectedChipText: {
    color: '#1e3c72',
    fontWeight: '600',
  },
  emergencyContactsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  addButton: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 20,
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  noContactsText: {
    color: 'rgba(255,255,255,0.7)',
    fontStyle: 'italic',
    marginBottom: 24,
    textAlign: 'center',
    fontSize: 14,
  },
  contactCard: {
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  contactInfo: {
    flex: 1,
  },
  contactName: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  contactDetail: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 14,
    marginBottom: 2,
  },
  contactActions: {
    flexDirection: 'row',
    gap: 8,
  },
  actionButton: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 16,
    width: 32,
    height: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },
  button: {
    marginTop: 32,
    backgroundColor: 'white',
    borderRadius: 25,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 8,
    },
    shadowOpacity: 0.2,
    shadowRadius: 20,
    elevation: 8,
  },
  buttonContent: {
    height: 60,
  },
  buttonLabel: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1e3c72',
  },
  dialog: {
    backgroundColor: 'white',
    borderRadius: 16,
  },
  dialogTitle: {
    color: '#1e3c72',
    fontWeight: '600',
  },
  dialogInput: {
    marginBottom: 16,
  },
}); 