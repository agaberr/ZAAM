import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, Platform, KeyboardAvoidingView, TouchableOpacity, Dimensions } from 'react-native';
import { Text, TextInput, Button, Switch, Modal, Portal, RadioButton, Chip, HelperText } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';
import { format } from 'date-fns';
import { reminderService, ReminderData } from '../services/reminderService';

const { width } = Dimensions.get('window');

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
          {/* Modern Header */}
          <View style={styles.header}>
            <TouchableOpacity style={styles.closeButton} onPress={onDismiss}>
              <Ionicons name="close" size={24} color="white" />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>
              {editReminder ? 'Edit Reminder' : 'New Reminder'}
            </Text>
            <View style={styles.headerSpacer} />
          </View>

          <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
            {error ? (
              <View style={styles.errorContainer}>
                <Ionicons name="alert-circle" size={20} color="#EF4444" />
                <Text style={styles.errorText}>{error}</Text>
              </View>
            ) : null}
            
            {/* Title Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Title</Text>
              <TextInput
                value={title}
                onChangeText={setTitle}
                style={styles.input}
                mode="outlined"
                placeholder="Enter reminder title"
                outlineColor="#E2E8F0"
                activeOutlineColor="#6366F1"
                theme={{
                  colors: {
                    primary: '#6366F1',
                    outline: '#E2E8F0',
                  }
                }}
              />
            </View>
            
            {/* Description Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.inputLabel}>Description (Optional)</Text>
              <TextInput
                value={description}
                onChangeText={setDescription}
                style={[styles.input, styles.textArea]}
                mode="outlined"
                placeholder="Add description or notes"
                multiline
                numberOfLines={3}
                outlineColor="#E2E8F0"
                activeOutlineColor="#6366F1"
                theme={{
                  colors: {
                    primary: '#6366F1',
                    outline: '#E2E8F0',
                  }
                }}
              />
            </View>
            
            {/* Date & Time Section */}
            <View style={styles.sectionContainer}>
              <Text style={styles.sectionTitle}>
                <Ionicons name="calendar" size={18} color="#6366F1" /> Date & Time
              </Text>
              
              <Text style={styles.subLabel}>Start</Text>
              <View style={styles.dateTimeRow}>
                <TouchableOpacity
                  style={styles.dateTimeButton}
                  onPress={() => setShowStartDatePicker(true)}
                >
                  <Ionicons name="calendar-outline" size={18} color="#64748B" />
                  <Text style={styles.dateTimeText}>
                    {format(startDate, 'MMM d, yyyy')}
                  </Text>
                </TouchableOpacity>
                
                <TouchableOpacity
                  style={styles.dateTimeButton}
                  onPress={() => setShowStartTimePicker(true)}
                >
                  <Ionicons name="time-outline" size={18} color="#64748B" />
                  <Text style={styles.dateTimeText}>
                    {format(startDate, 'h:mm a')}
                  </Text>
                </TouchableOpacity>
              </View>
              
              {/* End Date Toggle */}
              <View style={styles.toggleContainer}>
                <View style={styles.toggleLeft}>
                  <Ionicons name="flag-outline" size={18} color="#64748B" />
                  <Text style={styles.toggleLabel}>Include end time</Text>
                </View>
                <Switch
                  value={withEndDate}
                  onValueChange={setWithEndDate}
                  thumbColor={withEndDate ? '#6366F1' : '#F1F5F9'}
                  trackColor={{ false: '#E2E8F0', true: '#C7D2FE' }}
                />
              </View>
              
              {withEndDate && (
                <>
                  <Text style={styles.subLabel}>End</Text>
                  <View style={styles.dateTimeRow}>
                    <TouchableOpacity
                      style={styles.dateTimeButton}
                      onPress={() => setShowEndDatePicker(true)}
                    >
                      <Ionicons name="calendar-outline" size={18} color="#64748B" />
                      <Text style={styles.dateTimeText}>
                        {format(endDate, 'MMM d, yyyy')}
                      </Text>
                    </TouchableOpacity>
                    
                    <TouchableOpacity
                      style={styles.dateTimeButton}
                      onPress={() => setShowEndTimePicker(true)}
                    >
                      <Ionicons name="time-outline" size={18} color="#64748B" />
                      <Text style={styles.dateTimeText}>
                        {format(endDate, 'h:mm a')}
                      </Text>
                    </TouchableOpacity>
                  </View>
                </>
              )}
            </View>
            
            {/* Recurrence Section */}
            <View style={styles.sectionContainer}>
              <Text style={styles.sectionTitle}>
                <Ionicons name="repeat" size={18} color="#6366F1" /> Recurrence
              </Text>
              
              <View style={styles.toggleContainer}>
                <View style={styles.toggleLeft}>
                  <Ionicons name="refresh-outline" size={18} color="#64748B" />
                  <Text style={styles.toggleLabel}>Recurring reminder</Text>
                </View>
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
                  thumbColor={recurring ? '#6366F1' : '#F1F5F9'}
                  trackColor={{ false: '#E2E8F0', true: '#C7D2FE' }}
                />
              </View>
              
              {recurring && (
                <View style={styles.recurrenceOptions}>
                  <RadioButton.Group
                    onValueChange={value => setRecurrence(value)}
                    value={recurrence || ''}
                  >
                    {[
                      { value: 'daily', label: 'Daily', icon: 'sunny' },
                      { value: 'weekly', label: 'Weekly', icon: 'calendar' },
                      { value: 'monthly', label: 'Monthly', icon: 'calendar-outline' },
                      { value: 'yearly', label: 'Yearly', icon: 'calendar-clear' }
                    ].map((option) => (
                      <TouchableOpacity
                        key={option.value}
                        style={[
                          styles.recurrenceOption,
                          recurrence === option.value && styles.recurrenceOptionSelected
                        ]}
                        onPress={() => setRecurrence(option.value)}
                      >
                        <Ionicons 
                          name={option.icon as any} 
                          size={20} 
                          color={recurrence === option.value ? '#6366F1' : '#64748B'} 
                        />
                        <Text style={[
                          styles.recurrenceOptionText,
                          recurrence === option.value && styles.recurrenceOptionTextSelected
                        ]}>
                          {option.label}
                        </Text>
                        <RadioButton value={option.value} />
                      </TouchableOpacity>
                    ))}
                  </RadioButton.Group>
                </View>
              )}
            </View>
          </ScrollView>

          {/* Action Buttons */}
          <View style={styles.actionContainer}>
            <TouchableOpacity
              style={styles.cancelButton}
              onPress={onDismiss}
            >
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[
                styles.submitButton,
                (!title.trim() || loading) && styles.submitButtonDisabled
              ]}
              onPress={handleSubmit}
              disabled={!title.trim() || loading}
            >
              {loading ? (
                <Text style={styles.submitButtonText}>Saving...</Text>
              ) : (
                <>
                  <Ionicons name="checkmark" size={20} color="white" />
                  <Text style={styles.submitButtonText}>
                    {editReminder ? 'Update' : 'Create'}
                  </Text>
                </>
              )}
            </TouchableOpacity>
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
        </KeyboardAvoidingView>
      </Modal>
    </Portal>
  );
};

const styles = StyleSheet.create({
  modalContainer: {
    backgroundColor: 'white',
    margin: 16,
    borderRadius: 24,
    maxHeight: '90%',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 10,
    },
    shadowOpacity: 0.25,
    shadowRadius: 20,
    elevation: 10,
    overflow: 'hidden',
  },
  container: {
    flex: 1,
  },
  header: {
    backgroundColor: '#6366F1',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
    paddingTop: 20,
  },
  closeButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 20,
    padding: 8,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: 'white',
    flex: 1,
    textAlign: 'center',
    marginRight: 44, // Account for close button width
  },
  headerSpacer: {
    width: 44, // Same width as close button for center alignment
  },
  scrollView: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 24,
  },
  errorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FEF2F2',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#FECACA',
  },
  errorText: {
    fontSize: 14,
    color: '#EF4444',
    marginLeft: 8,
    fontWeight: '500',
    flex: 1,
  },
  inputContainer: {
    marginBottom: 20,
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#F8FAFC',
    borderRadius: 12,
    fontSize: 16,
  },
  textArea: {
    minHeight: 80,
    textAlignVertical: 'top',
  },
  sectionContainer: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
    borderWidth: 1,
    borderColor: '#F1F5F9',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
  },
  subLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 12,
    marginTop: 8,
  },
  dateTimeRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  dateTimeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F8FAFC',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    flex: 1,
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
  dateTimeText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
    marginLeft: 8,
  },
  toggleContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#F8FAFC',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderRadius: 12,
    marginVertical: 12,
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  toggleLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  toggleLabel: {
    fontSize: 16,
    fontWeight: '500',
    color: '#374151',
    marginLeft: 8,
  },
  recurrenceOptions: {
    marginTop: 12,
  },
  recurrenceOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderRadius: 12,
    marginBottom: 8,
    backgroundColor: '#F8FAFC',
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  recurrenceOptionSelected: {
    backgroundColor: '#EEF2FF',
    borderColor: '#6366F1',
    borderWidth: 2,
  },
  recurrenceOptionText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#374151',
    marginLeft: 12,
    flex: 1,
  },
  recurrenceOptionTextSelected: {
    color: '#6366F1',
    fontWeight: '600',
  },
  actionContainer: {
    flexDirection: 'row',
    gap: 12,
    paddingHorizontal: 20,
    paddingVertical: 20,
    backgroundColor: '#F8FAFC',
    borderTopWidth: 1,
    borderTopColor: '#E2E8F0',
  },
  cancelButton: {
    flex: 1,
    paddingVertical: 16,
    backgroundColor: 'white',
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 2,
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
  cancelButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#64748B',
  },
  submitButton: {
    flex: 1,
    flexDirection: 'row',
    paddingVertical: 16,
    backgroundColor: '#6366F1',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#6366F1',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
    gap: 8,
  },
  submitButtonDisabled: {
    backgroundColor: '#CBD5E1',
    shadowOpacity: 0,
    elevation: 0,
  },
  submitButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: 'white',
  },
});

export default ReminderForm; 