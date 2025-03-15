import React, { useState } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { TextInput, Button, Title, Chip, HelperText } from 'react-native-paper';
import { useAuth } from '../../context/AuthContext';

const UserDataScreen = ({ route }) => {
  const { email, password } = route.params;
  const { signup } = useAuth();
  
  const [fullName, setFullName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [phone, setPhone] = useState('');
  const [medications, setMedications] = useState([]);
  const [newMedication, setNewMedication] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAddMedication = () => {
    if (newMedication.trim()) {
      setMedications([...medications, newMedication.trim()]);
      setNewMedication('');
    }
  };

  const handleSubmit = async () => {
    if (!fullName || !age || !gender) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const userData = {
        full_name: fullName,
        age: parseInt(age),
        gender,
        contact_info: {
          phone,
          email,
        },
        emergency_contacts: [],
        preferences: {
          language: 'English',
          voice_type: 'Female',
          reminder_frequency: 'daily',
        },
      };

      await signup(userData, password);
    } catch (err) {
      setError('Failed to create account. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Title style={styles.title}>Tell Us About Yourself</Title>

      <TextInput
        label="Full Name *"
        value={fullName}
        onChangeText={setFullName}
        mode="outlined"
        style={styles.input}
      />

      <TextInput
        label="Age *"
        value={age}
        onChangeText={setAge}
        keyboardType="numeric"
        mode="outlined"
        style={styles.input}
      />

      <TextInput
        label="Gender *"
        value={gender}
        onChangeText={setGender}
        mode="outlined"
        style={styles.input}
      />

      <TextInput
        label="Phone Number"
        value={phone}
        onChangeText={setPhone}
        keyboardType="phone-pad"
        mode="outlined"
        style={styles.input}
      />

      <Title style={styles.sectionTitle}>Medications</Title>
      <View style={styles.medicationsContainer}>
        {medications.map((med, index) => (
          <Chip
            key={index}
            style={styles.chip}
            onClose={() => {
              setMedications(medications.filter((_, i) => i !== index));
            }}
          >
            {med}
          </Chip>
        ))}
      </View>

      <View style={styles.medicationInput}>
        <TextInput
          label="Add Medication"
          value={newMedication}
          onChangeText={setNewMedication}
          mode="outlined"
          style={[styles.input, { flex: 1 }]}
        />
        <Button
          mode="contained"
          onPress={handleAddMedication}
          style={styles.addButton}
        >
          Add
        </Button>
      </View>

      {error ? <HelperText type="error">{error}</HelperText> : null}

      <Button
        mode="contained"
        onPress={handleSubmit}
        style={styles.button}
        loading={loading}
        disabled={loading}
      >
        Complete Profile
      </Button>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    marginTop: 40,
    marginBottom: 20,
    textAlign: 'center',
  },
  sectionTitle: {
    fontSize: 18,
    marginTop: 10,
    marginBottom: 10,
  },
  input: {
    marginBottom: 15,
    backgroundColor: 'white',
  },
  medicationsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 10,
  },
  chip: {
    margin: 4,
  },
  medicationInput: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  addButton: {
    marginLeft: 10,
  },
  button: {
    marginTop: 20,
    marginBottom: 40,
    paddingVertical: 6,
  },
});

export default UserDataScreen;
