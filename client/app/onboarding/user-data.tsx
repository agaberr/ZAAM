import React, { useState } from 'react';
import { StyleSheet, View, ScrollView } from 'react-native';
import { TextInput, Button, Text, SegmentedButtons, List } from 'react-native-paper';
import { router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

type Gender = 'male' | 'female' | 'other';

export default function UserDataScreen() {
  const [name, setName] = useState('');
  const [gender, setGender] = useState<Gender>('male');
  const [age, setAge] = useState('');
  const [medications, setMedications] = useState('');
  const [medicalInfo, setMedicalInfo] = useState('');
  const [caregiverName, setCaregiverName] = useState('');
  const [caregiverPhone, setCaregiverPhone] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    try {
      setLoading(true);
      // TODO: Implement data submission to backend
      router.push('/dashboard');
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <StatusBar style="auto" />
      <View style={styles.header}>
        <Text variant="headlineMedium" style={styles.title}>Personal Information</Text>
        <Text variant="bodyLarge" style={styles.subtitle}>
          Help us personalize your experience
        </Text>
      </View>

      <View style={styles.form}>
        <TextInput
          label="Full Name"
          value={name}
          onChangeText={setName}
          mode="outlined"
          style={styles.input}
        />

        <Text variant="bodyMedium" style={styles.label}>Gender</Text>
        <SegmentedButtons
          value={gender}
          onValueChange={value => setGender(value as Gender)}
          buttons={[
            { value: 'male', label: 'Male' },
            { value: 'female', label: 'Female' },
            { value: 'other', label: 'Other' }
          ]}
          style={styles.segmentedButton}
        />

        <TextInput
          label="Age"
          value={age}
          onChangeText={setAge}
          mode="outlined"
          keyboardType="numeric"
          style={styles.input}
        />

        <TextInput
          label="Medications Used"
          value={medications}
          onChangeText={setMedications}
          mode="outlined"
          multiline
          numberOfLines={3}
          style={styles.input}
          placeholder="List your current medications..."
        />

        <TextInput
          label="Additional Medical Information"
          value={medicalInfo}
          onChangeText={setMedicalInfo}
          mode="outlined"
          multiline
          numberOfLines={4}
          style={styles.input}
          placeholder="Any relevant medical conditions or notes..."
        />

        <List.Accordion
          title="Caregiver's Information (Optional)"
          style={styles.accordion}
        >
          <TextInput
            label="Caregiver's Name"
            value={caregiverName}
            onChangeText={setCaregiverName}
            mode="outlined"
            style={styles.input}
          />

          <TextInput
            label="Caregiver's Phone"
            value={caregiverPhone}
            onChangeText={setCaregiverPhone}
            mode="outlined"
            keyboardType="phone-pad"
            style={styles.input}
          />
        </List.Accordion>

        <Button
          mode="contained"
          onPress={handleSubmit}
          style={styles.button}
          loading={loading}
          disabled={!name || !age}
        >
          Continue to Dashboard
        </Button>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#fff',
    padding: 20,
  },
  header: {
    marginTop: 60,
    marginBottom: 40,
  },
  title: {
    textAlign: 'center',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.7,
  },
  form: {
    gap: 16,
  },
  input: {
    marginBottom: 16,
  },
  label: {
    marginBottom: 8,
  },
  segmentedButton: {
    marginBottom: 16,
  },
  accordion: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    marginBottom: 16,
  },
  button: {
    marginTop: 24,
    marginBottom: 40,
  },
}); 