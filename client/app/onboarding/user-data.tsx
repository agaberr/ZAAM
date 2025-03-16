import React, { useState } from 'react';
import { StyleSheet, View, ScrollView } from 'react-native';
import { TextInput, Button, Text, IconButton, Chip } from 'react-native-paper';
import { router } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../context/AuthContext';

export default function UserDataScreen() {
  const [age, setAge] = useState('');
  const [relation, setRelation] = useState('');
  const [selectedCondition, setSelectedCondition] = useState('early');
  const [loading, setLoading] = useState(false);
  
  const { completeOnboarding } = useAuth();

  const handleComplete = async () => {
    setLoading(true);
    
    try {
      // In a real app, you would save this data to your backend
      await completeOnboarding();
      // No need to navigate here - the auth context will handle it
    } catch (error) {
      console.error('Failed to complete onboarding:', error);
    } finally {
      setLoading(false);
    }
  };

  const skipOnboarding = async () => {
    setLoading(true);
    
    try {
      await completeOnboarding();
    } catch (error) {
      console.error('Failed to skip onboarding:', error);
    } finally {
      setLoading(false);
    }
  };

  const goBack = () => {
    router.back();
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
          <Text variant="headlineMedium" style={styles.title}>Patient Information</Text>
          <Text variant="bodyLarge" style={styles.subtitle}>
            Help us personalize the experience for your loved one
          </Text>
          
          <View style={styles.form}>
            <TextInput
              label="Patient's Age"
              value={age}
              onChangeText={setAge}
              mode="outlined"
              style={styles.input}
              keyboardType="number-pad"
              outlineColor="rgba(255,255,255,0.3)"
              activeOutlineColor="white"
              textColor="white"
              theme={{ colors: { onSurfaceVariant: 'white' } }}
            />
            
            <TextInput
              label="Your Relationship to Patient"
              value={relation}
              onChangeText={setRelation}
              mode="outlined"
              style={styles.input}
              placeholder="e.g. Son, Daughter, Caregiver"
              outlineColor="rgba(255,255,255,0.3)"
              activeOutlineColor="white"
              textColor="white"
              theme={{ colors: { onSurfaceVariant: 'white' } }}
            />
            
            <Text style={styles.sectionTitle}>Alzheimer's Stage</Text>
            <View style={styles.chipContainer}>
              <Chip
                selected={selectedCondition === 'early'}
                onPress={() => setSelectedCondition('early')}
                style={[
                  styles.chip,
                  selectedCondition === 'early' && styles.selectedChip
                ]}
                textStyle={[
                  styles.chipText,
                  selectedCondition === 'early' && styles.selectedChipText
                ]}
              >
                Early Stage
              </Chip>
              
              <Chip
                selected={selectedCondition === 'middle'}
                onPress={() => setSelectedCondition('middle')}
                style={[
                  styles.chip,
                  selectedCondition === 'middle' && styles.selectedChip
                ]}
                textStyle={[
                  styles.chipText,
                  selectedCondition === 'middle' && styles.selectedChipText
                ]}
              >
                Middle Stage
              </Chip>
              
              <Chip
                selected={selectedCondition === 'late'}
                onPress={() => setSelectedCondition('late')}
                style={[
                  styles.chip,
                  selectedCondition === 'late' && styles.selectedChip
                ]}
                textStyle={[
                  styles.chipText,
                  selectedCondition === 'late' && styles.selectedChipText
                ]}
              >
                Late Stage
              </Chip>
            </View>
            
            <Button
              mode="contained"
              onPress={handleComplete}
              style={styles.button}
              contentStyle={styles.buttonContent}
              labelStyle={styles.buttonLabel}
              loading={loading}
              disabled={loading}
            >
              Complete Setup
            </Button>
            
            <Button
              mode="text"
              onPress={skipOnboarding}
              style={styles.skipButton}
              labelStyle={styles.skipButtonLabel}
              disabled={loading}
            >
              Skip for now
            </Button>
          </View>
        </View>
      </ScrollView>
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
    justifyContent: 'center',
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
    marginBottom: 16,
    backgroundColor: 'transparent',
  },
  sectionTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: '500',
    marginTop: 8,
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
  skipButton: {
    marginTop: 16,
  },
  skipButtonLabel: {
    color: 'white',
  },
}); 