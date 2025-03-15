import React, { useState } from 'react';
import { StyleSheet, View, ScrollView } from 'react-native';
import { Text, FAB, Card, List, IconButton, Button, Portal, Modal, TextInput } from 'react-native-paper';
import { router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import DateTimePicker from '@react-native-community/datetimepicker';

type Reminder = {
  id: string;
  title: string;
  type: 'medication' | 'appointment' | 'activity';
  datetime: Date;
  notes?: string;
};

export default function RemindersScreen() {
  const [reminders, setReminders] = useState<Reminder[]>([
    {
      id: '1',
      title: 'Take Aspirin',
      type: 'medication',
      datetime: new Date(2024, 2, 20, 9, 0),
      notes: '100mg with water',
    },
    {
      id: '2',
      title: 'Doctor Appointment',
      type: 'appointment',
      datetime: new Date(2024, 2, 21, 14, 30),
      notes: 'Regular checkup',
    },
  ]);

  const [visible, setVisible] = useState(false);
  const [newReminder, setNewReminder] = useState<Partial<Reminder>>({
    type: 'medication',
    datetime: new Date(),
  });
  const [showDatePicker, setShowDatePicker] = useState(false);

  const showModal = () => setVisible(true);
  const hideModal = () => {
    setVisible(false);
    setNewReminder({ type: 'medication', datetime: new Date() });
  };

  const handleAddReminder = () => {
    if (newReminder.title) {
      setReminders([
        ...reminders,
        {
          id: Date.now().toString(),
          title: newReminder.title,
          type: newReminder.type || 'medication',
          datetime: newReminder.datetime || new Date(),
          notes: newReminder.notes,
        },
      ]);
      hideModal();
    }
  };

  const handleDeleteReminder = (id: string) => {
    setReminders(reminders.filter(reminder => reminder.id !== id));
  };

  const formatDate = (date: Date) => {
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  };

  return (
    <View style={styles.container}>
      <StatusBar style="auto" />
      
      {/* Header */}
      <View style={styles.header}>
        <IconButton icon="arrow-left" size={24} onPress={() => router.back()} />
        <Text variant="headlineMedium" style={styles.title}>Reminders</Text>
        <IconButton 
          icon="calendar-sync" 
          size={24} 
          onPress={() => console.log('Sync with Google Calendar')} 
        />
      </View>

      {/* Reminders List */}
      <ScrollView style={styles.content}>
        {reminders.map((reminder) => (
          <Card key={reminder.id} style={styles.card}>
            <Card.Content>
              <View style={styles.reminderHeader}>
                <View style={styles.reminderInfo}>
                  <Text variant="titleMedium">{reminder.title}</Text>
                  <Text variant="bodyMedium" style={styles.datetime}>
                    {formatDate(reminder.datetime)}
                  </Text>
                </View>
                <IconButton
                  icon="delete"
                  size={20}
                  onPress={() => handleDeleteReminder(reminder.id)}
                />
              </View>
              {reminder.notes && (
                <Text variant="bodyMedium" style={styles.notes}>
                  {reminder.notes}
                </Text>
              )}
            </Card.Content>
          </Card>
        ))}
      </ScrollView>

      {/* Add Reminder FAB */}
      <FAB
        icon="plus"
        style={styles.fab}
        onPress={showModal}
        label="Add Reminder"
      />

      {/* Add Reminder Modal */}
      <Portal>
        <Modal
          visible={visible}
          onDismiss={hideModal}
          contentContainerStyle={styles.modalContent}
        >
          <Text variant="titleLarge" style={styles.modalTitle}>Add Reminder</Text>
          
          <TextInput
            label="Title"
            value={newReminder.title || ''}
            onChangeText={(text) => setNewReminder({ ...newReminder, title: text })}
            mode="outlined"
            style={styles.input}
          />

          <List.Accordion
            title="Type"
            left={props => <List.Icon {...props} icon="folder" />}
            expanded={true}
          >
            <List.Item
              title="Medication"
              left={props => <List.Icon {...props} icon="pill" />}
              onPress={() => setNewReminder({ ...newReminder, type: 'medication' })}
              right={() => newReminder.type === 'medication' ? <List.Icon icon="check" /> : null}
            />
            <List.Item
              title="Appointment"
              left={props => <List.Icon {...props} icon="calendar" />}
              onPress={() => setNewReminder({ ...newReminder, type: 'appointment' })}
              right={() => newReminder.type === 'appointment' ? <List.Icon icon="check" /> : null}
            />
            <List.Item
              title="Activity"
              left={props => <List.Icon {...props} icon="run" />}
              onPress={() => setNewReminder({ ...newReminder, type: 'activity' })}
              right={() => newReminder.type === 'activity' ? <List.Icon icon="check" /> : null}
            />
          </List.Accordion>

          <Button
            mode="outlined"
            onPress={() => setShowDatePicker(true)}
            style={styles.dateButton}
          >
            {formatDate(newReminder.datetime || new Date())}
          </Button>

          {showDatePicker && (
            <DateTimePicker
              value={newReminder.datetime || new Date()}
              mode="datetime"
              onChange={(event, date) => {
                setShowDatePicker(false);
                if (date) {
                  setNewReminder({ ...newReminder, datetime: date });
                }
              }}
            />
          )}

          <TextInput
            label="Notes"
            value={newReminder.notes || ''}
            onChangeText={(text) => setNewReminder({ ...newReminder, notes: text })}
            mode="outlined"
            multiline
            numberOfLines={3}
            style={styles.input}
          />

          <View style={styles.modalButtons}>
            <Button onPress={hideModal} style={styles.modalButton}>Cancel</Button>
            <Button
              mode="contained"
              onPress={handleAddReminder}
              style={styles.modalButton}
              disabled={!newReminder.title}
            >
              Add
            </Button>
          </View>
        </Modal>
      </Portal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 60,
    paddingBottom: 20,
    paddingHorizontal: 16,
    backgroundColor: '#fff',
  },
  title: {
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  card: {
    marginBottom: 12,
    borderRadius: 12,
  },
  reminderHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  reminderInfo: {
    flex: 1,
  },
  datetime: {
    opacity: 0.7,
    marginTop: 4,
  },
  notes: {
    marginTop: 8,
    opacity: 0.7,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
  },
  modalContent: {
    backgroundColor: 'white',
    padding: 20,
    margin: 20,
    borderRadius: 12,
  },
  modalTitle: {
    marginBottom: 20,
    textAlign: 'center',
    fontWeight: 'bold',
  },
  input: {
    marginBottom: 16,
  },
  dateButton: {
    marginVertical: 16,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 8,
    marginTop: 16,
  },
  modalButton: {
    minWidth: 100,
  },
}); 