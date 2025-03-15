import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, RefreshControl } from 'react-native';
import { Card, Title, FAB, List, Divider, Chip, Button, Dialog, Portal, TextInput } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';

const RemindersScreen = () => {
  const [refreshing, setRefreshing] = useState(false);
  const [reminders, setReminders] = useState([]);
  const [dialogVisible, setDialogVisible] = useState(false);
  const [reminderType, setReminderType] = useState('medication');
  const [reminderName, setReminderName] = useState('');
  const [reminderDosage, setReminderDosage] = useState('');
  const [reminderTime, setReminderTime] = useState(new Date());
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [reminderRecurrence, setReminderRecurrence] = useState('daily');

  // Fetch reminders
  const fetchReminders = async () => {
    setRefreshing(true);
    try {
      // In a real app, you would fetch data from your API
      // For demo purposes, we'll use mock data
      setReminders([
        {
          id: 1,
          type: 'medication',
          name: 'Aspirin',
          dosage: '100mg',
          time: '08:00 AM',
          recurrence: 'daily',
          status: 'pending',
        },
        {
          id: 2,
          type: 'medication',
          name: 'Vitamin D',
          dosage: '1000 IU',
          time: '10:00 AM',
          recurrence: 'daily',
          status: 'completed',
        },
        {
          id: 3,
          type: 'hydration',
          name: 'Drink Water',
          time: '12:00 PM',
          recurrence: 'every 2 hours',
          status: 'pending',
        },
        {
          id: 4,
          type: 'appointment',
          name: 'Doctor Visit',
          time: '3:00 PM',
          date: 'Tomorrow',
          recurrence: 'once',
          status: 'pending',
        },
      ]);
    } catch (error) {
      console.error('Error fetching reminders', error);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchReminders();
  }, []);

  const onRefresh = () => {
    fetchReminders();
  };

  const handleAddReminder = () => {
    // Reset form fields
    setReminderType('medication');
    setReminderName('');
    setReminderDosage('');
    setReminderTime(new Date());
    setReminderRecurrence('daily');
    setDialogVisible(true);
  };

  const handleSaveReminder = () => {
    // In a real app, you would send this to your API
    const newReminder = {
      id: reminders.length + 1,
      type: reminderType,
      name: reminderName,
      ...(reminderType === 'medication' && { dosage: reminderDosage }),
      time: reminderTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      recurrence: reminderRecurrence,
      status: 'pending',
    };
    
    setReminders([...reminders, newReminder]);
    setDialogVisible(false);
  };

  const handleTimeChange = (event, selectedTime) => {
    setShowTimePicker(false);
    if (selectedTime) {
      setReminderTime(selectedTime);
    }
  };

  const getReminderIcon = (type) => {
    switch (type) {
      case 'medication':
        return 'pill';
      case 'hydration':
        return 'cup-water';
      case 'appointment':
        return 'calendar-clock';
      case 'exercise':
        return 'walk';
      default:
        return 'bell';
    }
  };

  const toggleReminderStatus = (id) => {
    setReminders(reminders.map(reminder => 
      reminder.id === id 
        ? { ...reminder, status: reminder.status === 'completed' ? 'pending' : 'completed' } 
        : reminder
    ));
  };

  return (
    <View style={styles.container}>
      <ScrollView
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        <Card style={styles.card}>
          <Card.Content>
            <Title style={styles.cardTitle}>Today's Reminders</Title>
            
            {reminders.length === 0 ? (
              <Paragraph style={styles.emptyText}>No reminders for today</Paragraph>
            ) : (
              reminders.map((reminder) => (
                <View key={reminder.id}>
                  <List.Item
                    title={reminder.name}
                    description={
                      reminder.type === 'medication' 
                        ? `${reminder.dosage} at ${reminder.time} (${reminder.recurrence})`
                        : `${reminder.time} (${reminder.recurrence})`
                    }
                    left={props => <List.Icon {...props} icon={getReminderIcon(reminder.type)} />}
                    right={() => (
                      <Button 
                        mode={reminder.status === 'completed' ? 'contained' : 'outlined'}
                        onPress={() => toggleReminderStatus(reminder.id)}
                        style={reminder.status === 'completed' ? styles.completedButton : styles.pendingButton}
                        labelStyle={reminder.status === 'completed' ? styles.completedButtonLabel : {}}
                      >
                        {reminder.status === 'completed' ? 'Done' : 'Mark Done'}
                      </Button>
                    )}
                  />
                  {reminder.id !== reminders.length && <Divider />}
                </View>
              ))
            )}
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Content>
            <Title style={styles.cardTitle}>Reminder Categories</Title>
            <View style={styles.categoriesContainer}>
              <Chip 
                icon="pill" 
                style={styles.categoryChip} 
                onPress={() => {/* Filter by medication */}}
              >
                Medications
              </Chip>
              <Chip 
                icon="cup-water" 
                style={styles.categoryChip} 
                onPress={() => {/* Filter by hydration */}}
              >
                Hydration
              </Chip>
              <Chip 
                icon="calendar-clock" 
                style={styles.categoryChip} 
                onPress={() => {/* Filter by appointments */}}
              >
                Appointments
              </Chip>
              <Chip 
                icon="walk" 
                style={styles.categoryChip} 
                onPress={() => {/* Filter by exercise */}}
              >
                Exercise
              </Chip>
            </View>
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Content>
            <Title style={styles.cardTitle}>Google Calendar Sync</Title>
            <Paragraph>
              Sync your reminders with Google Calendar to access them on all your devices.
            </Paragraph>
            <Button 
              mode="outlined" 
              icon="google" 
              onPress={() => {/* Implement Google Calendar sync */}}
              style={styles.syncButton}
            >
              Sync with Google Calendar
            </Button>
          </Card.Content>
        </Card>

        <View style={styles.spacer} />
      </ScrollView>

      <FAB
        style={styles.fab}
        icon="plus"
        onPress={handleAddReminder}
        label="Add Reminder"
      />

      <Portal>
        <Dialog visible={dialogVisible} onDismiss={() => setDialogVisible(false)}>
          <Dialog.Title>Add New Reminder</Dialog.Title>
          <Dialog.Content>
            <View style={styles.dialogContent}>
              <Title style={styles.sectionTitle}>Reminder Type</Title>
              <View style={styles.typeContainer}>
                <Chip 
                  icon="pill" 
                  selected={reminderType === 'medication'}
                  onPress={() => setReminderType('medication')}
                  style={styles.typeChip}
                >
                  Medication
                </Chip>
                <Chip 
                  icon="cup-water" 
                  selected={reminderType === 'hydration'}
                  onPress={() => setReminderType('hydration')}
                  style={styles.typeChip}
                >
                  Hydration
                </Chip>
                <Chip 
                  icon="calendar-clock" 
                  selected={reminderType === 'appointment'}
                  onPress={() => setReminderType('appointment')}
                  style={styles.typeChip}
                >
                  Appointment
                </Chip>
                <Chip 
                  icon="walk" 
                  selected={reminderType === 'exercise'}
                  onPress={() => setReminderType('exercise')}
                  style={styles.typeChip}
                >
                  Exercise
                </Chip>
              </View>

              <TextInput
                label="Reminder Name"
                value={reminderName}
                onChangeText={setReminderName}
                mode="outlined"
                style={styles.input}
              />

              {reminderType === 'medication' && (
                <TextInput
                  label="Dosage"
                  value={reminderDosage}
                  onChangeText={setReminderDosage}
                  mode="outlined"
                  style={styles.input}
                />
              )}

              <Title style={styles.sectionTitle}>Time</Title>
              <Button 
                mode="outlined" 
                onPress={() => setShowTimePicker(true)}
                style={styles.timeButton}
              >
                {reminderTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </Button>

              {showTimePicker && (
                <DateTimePicker
                  value={reminderTime}
                  mode="time"
                  is24Hour={false}
                  display="default"
                  onChange={handleTimeChange}
                />
              )}

              <Title style={styles.sectionTitle}>Recurrence</Title>
              <View style={styles.recurrenceContainer}>
                <Chip 
                  selected={reminderRecurrence === 'once'}
                  onPress={() => setReminderRecurrence('once')}
                  style={styles.recurrenceChip}
                >
                  Once
                </Chip>
                <Chip 
                  selected={reminderRecurrence === 'daily'}
                  onPress={() => setReminderRecurrence('daily')}
                  style={styles.recurrenceChip}
                >
                  Daily
                </Chip>
                <Chip 
                  selected={reminderRecurrence === 'weekly'}
                  onPress={() => setReminderRecurrence('weekly')}
                  style={styles.recurrenceChip}
                >
                  Weekly
                </Chip>
              </View>
            </View>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setDialogVisible(false)}>Cancel</Button>
            <Button onPress={handleSaveReminder}>Save</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  card: {
    margin: 16,
    marginTop: 8,
    marginBottom: 8,
  },
  cardTitle: {
    fontSize: 18,
    marginBottom: 10,
  },
  emptyText: {
    textAlign: 'center',
    marginTop: 20,
    marginBottom: 20,
    fontStyle: 'italic',
  },
  categoriesContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 10,
  },
  categoryChip: {
    margin: 4,
  },
  syncButton: {
    marginTop: 15,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: '#6200ee',
  },
  completedButton: {
    backgroundColor: '#4CAF50',
  },
  completedButtonLabel: {
    color: 'white',
  },
  pendingButton: {},
  dialogContent: {
    marginTop: 10,
  },
  sectionTitle: {
    fontSize: 16,
    marginTop: 15,
    marginBottom: 10,
  },
  typeContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 15,
  },
  typeChip: {
    margin: 4,
  },
  input: {
    marginBottom: 10,
  },
  timeButton: {
    marginTop: 5,
    marginBottom: 10,
  },
  recurrenceContainer: {
    flexDirection: 'row',
    marginTop: 5,
  },
  recurrenceChip: {
    margin: 4,
  },
  spacer: {
    height: 80,
  },
});

export default RemindersScreen;
