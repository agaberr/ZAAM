import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, Platform, KeyboardAvoidingView } from 'react-native';
import { Text, TextInput, Button, Switch, Modal, Portal, RadioButton, Chip, HelperText } from 'react-native-paper';
import DateTimePicker from '@react-native-community/datetimepicker';
import { format } from 'date-fns';
import { reminderService, ReminderData } from '../services/reminderService';

interface ReminderFormProps {
  visible: boolean;
  onDismiss: () => void;
  onSuccess: () => void;
  editReminder?: ReminderData;
}

const ReminderForm: React.FC<ReminderFormProps> = ({
  visible,
  onDismiss,
  onSuccess,
  editReminder
}) => {
  // Form state
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [startDate, setStartDate] = useState(new Date());
  const [endDate, setEndDate] = useState(new Date(new Date().getTime() + 60 * 60 * 1000)); // +1 hour
  const [recurrence, setRecurrence] = useState<string | null>(null);
  const [recurring, setRecurring] = useState(false);
  const [withEndDate, setWithEndDate] = useState(true);
  const [showRecurrenceOptions, setShowRecurrenceOptions] = useState(false);
  
  // Date pickers
  const [showStartDatePicker, setShowStartDatePicker] = useState(false);
  const [showStartTimePicker, setShowStartTimePicker] = useState(false);
  const [showEndDatePicker, setShowEndDatePicker] = useState(false);
  const [showEndTimePicker, setShowEndTimePicker] = useState(false);
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Load edit data if provided
  useEffect(() => {
    if (editReminder) {
      setTitle(editReminder.title);
      setDescription(editReminder.description || '');
      
      if (editReminder.start_time) {
        setStartDate(new Date(editReminder.start_time));
      }
      
      if (editReminder.end_time) {
        setEndDate(new Date(editReminder.end_time));
        setWithEndDate(true);
      } else {
        setWithEndDate(false);
      }
      
      if (editReminder.recurrence) {
        setRecurring(true);
        setRecurrence(editReminder.recurrence);
      } else {
        setRecurring(false);
        setRecurrence(null);
      }
    }
  }, [editReminder]);
  
  // Handle date/time selection
  const onStartDateChange = (event: any, selectedDate?: Date) => {
    setShowStartDatePicker(false);
    if (selectedDate) {
      const newDate = new Date(startDate);
      newDate.setFullYear(selectedDate.getFullYear());
      newDate.setMonth(selectedDate.getMonth());
      newDate.setDate(selectedDate.getDate());
      setStartDate(newDate);
      
      // Update end date to be at least 1 hour after start date
      const minEndDate = new Date(newDate.getTime() + 60 * 60 * 1000);
      if (endDate < minEndDate) {
        setEndDate(minEndDate);
      }
    }
  };
  
  const onStartTimeChange = (event: any, selectedTime?: Date) => {
    setShowStartTimePicker(false);
    if (selectedTime) {
      const newDate = new Date(startDate);
      newDate.setHours(selectedTime.getHours());
      newDate.setMinutes(selectedTime.getMinutes());
      setStartDate(newDate);
      
      // Update end date to be at least 1 hour after start date
      const minEndDate = new Date(newDate.getTime() + 60 * 60 * 1000);
      if (endDate < minEndDate) {
        setEndDate(minEndDate);
      }
    }
  };
  
  const onEndDateChange = (event: any, selectedDate?: Date) => {
    setShowEndDatePicker(false);
    if (selectedDate) {
      const newDate = new Date(endDate);
      newDate.setFullYear(selectedDate.getFullYear());
      newDate.setMonth(selectedDate.getMonth());
      newDate.setDate(selectedDate.getDate());
      
      // Ensure end date is not before start date
      if (newDate >= startDate) {
        setEndDate(newDate);
      } else {
        setEndDate(new Date(startDate.getTime() + 60 * 60 * 1000));
      }
    }
  };
  
  const onEndTimeChange = (event: any, selectedTime?: Date) => {
    setShowEndTimePicker(false);
    if (selectedTime) {
      const newDate = new Date(endDate);
      newDate.setHours(selectedTime.getHours());
      newDate.setMinutes(selectedTime.getMinutes());
      
      // Ensure end time is not before start time on the same day
      if (newDate.toDateString() === startDate.toDateString() && newDate <= startDate) {
        setEndDate(new Date(startDate.getTime() + 60 * 60 * 1000));
      } else {
        setEndDate(newDate);
      }
    }
  };
  
  // Reset form
  const resetForm = () => {
    setTitle('');
    setDescription('');
    setStartDate(new Date());
    setEndDate(new Date(new Date().getTime() + 60 * 60 * 1000));
    setRecurrence(null);
    setRecurring(false);
    setWithEndDate(true);
    setError('');
  };
  
  // Submit form
  const handleSubmit = async () => {
    try {
      setLoading(true);
      setError('');
      
      // Validate inputs
      if (!title.trim()) {
        setError('Please enter a title');
        setLoading(false);
        return;
      }
      
      // Prepare reminder data
      const reminderData: ReminderData = {
        title: title.trim(),
        description: description.trim() || undefined,
        start_time: startDate.toISOString(),
        end_time: withEndDate ? endDate.toISOString() : undefined,
        recurrence: recurring ? (recurrence as any) : null,
      };
      
      if (editReminder?._id) {
        // Update existing reminder
        await reminderService.updateReminder(editReminder._id, reminderData);
      } else {
        // Create new reminder
        await reminderService.createReminder(reminderData);
      }
      
      // Reset and close
      resetForm();
      setLoading(false);
      onSuccess();
      onDismiss();
    } catch (error) {
      console.error('Error saving reminder:', error);
      setError('Failed to save reminder. Please try again.');
      setLoading(false);
    }
  };
  
  return (
    <Portal>
      <Modal
        visible={visible}
        onDismiss={onDismiss}
        contentContainerStyle={styles.modalContainer}
      >
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.container}
        >
          <ScrollView style={styles.scrollView}>
            <Text style={styles.title}>
              {editReminder ? 'Edit Reminder' : 'Create New Reminder'}
            </Text>
            
            {error ? (
              <HelperText type="error" visible={!!error}>
                {error}
              </HelperText>
            ) : null}
            
            <TextInput
              label="Title"
              value={title}
              onChangeText={setTitle}
              style={styles.input}
              mode="outlined"
            />
            
            <TextInput
              label="Description (optional)"
              value={description}
              onChangeText={setDescription}
              style={styles.input}
              mode="outlined"
              multiline
              numberOfLines={3}
            />
            
            <Text style={styles.sectionTitle}>Start Date & Time</Text>
            <View style={styles.dateRow}>
              <Button
                mode="outlined"
                onPress={() => setShowStartDatePicker(true)}
                style={styles.dateButton}
              >
                {format(startDate, 'MMM d, yyyy')}
              </Button>
              
              <Button
                mode="outlined"
                onPress={() => setShowStartTimePicker(true)}
                style={styles.dateButton}
              >
                {format(startDate, 'h:mm a')}
              </Button>
            </View>
            
            <View style={styles.switchRow}>
              <Text>Include end date/time</Text>
              <Switch
                value={withEndDate}
                onValueChange={setWithEndDate}
              />
            </View>
            
            {withEndDate && (
              <>
                <Text style={styles.sectionTitle}>End Date & Time</Text>
                <View style={styles.dateRow}>
                  <Button
                    mode="outlined"
                    onPress={() => setShowEndDatePicker(true)}
                    style={styles.dateButton}
                  >
                    {format(endDate, 'MMM d, yyyy')}
                  </Button>
                  
                  <Button
                    mode="outlined"
                    onPress={() => setShowEndTimePicker(true)}
                    style={styles.dateButton}
                  >
                    {format(endDate, 'h:mm a')}
                  </Button>
                </View>
              </>
            )}
            
            <View style={styles.switchRow}>
              <Text>Recurring reminder</Text>
              <Switch
                value={recurring}
                onValueChange={val => {
                  setRecurring(val);
                  if (val) {
                    setShowRecurrenceOptions(true);
                  } else {
                    setRecurrence(null);
                  }
                }}
              />
            </View>
            
            {recurring && (
              <View style={styles.recurrenceContainer}>
                <Text style={styles.sectionTitle}>Recurrence</Text>
                <RadioButton.Group
                  onValueChange={value => setRecurrence(value)}
                  value={recurrence || ''}
                >
                  <View style={styles.radioRow}>
                    <RadioButton value="daily" />
                    <Text>Daily</Text>
                  </View>
                  <View style={styles.radioRow}>
                    <RadioButton value="weekly" />
                    <Text>Weekly</Text>
                  </View>
                  <View style={styles.radioRow}>
                    <RadioButton value="monthly" />
                    <Text>Monthly</Text>
                  </View>
                  <View style={styles.radioRow}>
                    <RadioButton value="yearly" />
                    <Text>Yearly</Text>
                  </View>
                </RadioButton.Group>
              </View>
            )}
            
            <View style={styles.buttonRow}>
              <Button
                mode="outlined"
                onPress={onDismiss}
                style={styles.button}
              >
                Cancel
              </Button>
              
              <Button
                mode="contained"
                onPress={handleSubmit}
                loading={loading}
                disabled={loading || !title.trim()}
                style={styles.button}
              >
                {editReminder ? 'Update' : 'Create'}
              </Button>
            </View>
            
            {/* Date & Time Pickers */}
            {showStartDatePicker && (
              <DateTimePicker
                value={startDate}
                mode="date"
                display="default"
                onChange={onStartDateChange}
              />
            )}
            
            {showStartTimePicker && (
              <DateTimePicker
                value={startDate}
                mode="time"
                display="default"
                onChange={onStartTimeChange}
              />
            )}
            
            {showEndDatePicker && (
              <DateTimePicker
                value={endDate}
                mode="date"
                display="default"
                onChange={onEndDateChange}
                minimumDate={startDate}
              />
            )}
            
            {showEndTimePicker && (
              <DateTimePicker
                value={endDate}
                mode="time"
                display="default"
                onChange={onEndTimeChange}
              />
            )}
          </ScrollView>
        </KeyboardAvoidingView>
      </Modal>
    </Portal>
  );
};

const styles = StyleSheet.create({
  modalContainer: {
    backgroundColor: 'white',
    margin: 20,
    borderRadius: 10,
    maxHeight: '80%',
  },
  container: {
    flex: 1,
  },
  scrollView: {
    padding: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  input: {
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '500',
    marginTop: 12,
    marginBottom: 8,
  },
  dateRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  dateButton: {
    flex: 1,
    marginHorizontal: 4,
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginVertical: 12,
  },
  recurrenceContainer: {
    marginTop: 8,
    marginBottom: 16,
  },
  radioRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 24,
    marginBottom: 12,
  },
  button: {
    flex: 1,
    marginHorizontal: 4,
  },
});

export default ReminderForm; 