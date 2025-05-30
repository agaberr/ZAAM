import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, Platform, KeyboardAvoidingView, TouchableOpacity, Dimensions } from 'react-native';
import { Text, TextInput, Button, Switch, Modal, Portal, RadioButton, Chip, HelperText, Card } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { format, addDays, startOfToday, addHours } from 'date-fns';
import { reminderService, ReminderData } from '../services/reminderService';

const { width } = Dimensions.get('window');

interface ReminderFormProps {
  visible: boolean;
  onDismiss: () => void;
  onSuccess: () => void;
  editReminder?: ReminderData;
}

// Quick time presets for easier selection
const TIME_PRESETS = [
  { label: '9:00 AM', hour: 9, minute: 0 },
  { label: '12:00 PM', hour: 12, minute: 0 },
  { label: '2:00 PM', hour: 14, minute: 0 },
  { label: '6:00 PM', hour: 18, minute: 0 },
  { label: '8:00 PM', hour: 20, minute: 0 },
];

// Quick date presets
const DATE_PRESETS = [
  { label: 'Today', offset: 0 },
  { label: 'Tomorrow', offset: 1 },
  { label: 'In 2 days', offset: 2 },
  { label: 'Next week', offset: 7 },
];

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
  const [withEndDate, setWithEndDate] = useState(false); // Changed default to false for simpler UX
  const [showRecurrenceOptions, setShowRecurrenceOptions] = useState(false);
  
  // Date pickers
  const [showStartDatePicker, setShowStartDatePicker] = useState(false);
  const [showStartTimePicker, setShowStartTimePicker] = useState(false);
  const [showEndDatePicker, setShowEndDatePicker] = useState(false);
  const [showEndTimePicker, setShowEndTimePicker] = useState(false);
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimePreset, setSelectedTimePreset] = useState<string | null>(null);
  const [selectedDatePreset, setSelectedDatePreset] = useState<string | null>(null);

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
    } else {
      // Reset form for new reminder
      resetForm();
    }
  }, [editReminder, visible]);

  // Quick date selection
  const handleDatePresetSelect = (preset: { label: string; offset: number }) => {
    const newDate = addDays(startOfToday(), preset.offset);
    const updatedStartDate = new Date(startDate);
    updatedStartDate.setFullYear(newDate.getFullYear());
    updatedStartDate.setMonth(newDate.getMonth());
    updatedStartDate.setDate(newDate.getDate());
    
    setStartDate(updatedStartDate);
    setSelectedDatePreset(preset.label);
    
    // Update end date accordingly
    const minEndDate = new Date(updatedStartDate.getTime() + 60 * 60 * 1000);
    if (withEndDate && endDate < minEndDate) {
      setEndDate(minEndDate);
    }
  };

  // Quick time selection
  const handleTimePresetSelect = (preset: { label: string; hour: number; minute: number }) => {
    const newStartDate = new Date(startDate);
    newStartDate.setHours(preset.hour);
    newStartDate.setMinutes(preset.minute);
    setStartDate(newStartDate);
    setSelectedTimePreset(preset.label);
    
    // Update end date to be 1 hour later
    if (withEndDate) {
      const newEndDate = addHours(newStartDate, 1);
      setEndDate(newEndDate);
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
    setWithEndDate(false);
    setError('');
    setSelectedTimePreset(null);
    setSelectedDatePreset(null);
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
      
      console.log('[DEBUG] Form submitting reminder:', reminderData);
      console.log('[DEBUG] Start date object:', startDate);
      console.log('[DEBUG] Start time ISO:', startDate.toISOString());
      if (withEndDate) {
        console.log('[DEBUG] End date object:', endDate);
        console.log('[DEBUG] End time ISO:', endDate.toISOString());
      }
      
      if (editReminder?._id) {
        // Update existing reminder
        console.log('[DEBUG] Updating existing reminder:', editReminder._id);
        await reminderService.updateReminder(editReminder._id, reminderData);
        console.log('[DEBUG] Reminder updated successfully');
      } else {
        // Create new reminder
        console.log('[DEBUG] Creating new reminder');
        const result = await reminderService.createReminder(reminderData);
        console.log('[DEBUG] Reminder created successfully:', result);
      }
      
      // Reset and close
      console.log('[DEBUG] Resetting form and calling success callback');
      resetForm();
      setLoading(false);
      onSuccess();
      onDismiss();
      console.log('[DEBUG] Form submission completed');
    } catch (error) {
      console.error('[DEBUG] Error saving reminder:', error);
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
          style={styles.keyboardAvoidingView}
        >
          <ScrollView 
            style={styles.scrollView}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
          >
            {/* Header */}
            <View style={styles.header}>
              <Text style={styles.modalTitle}>
                {editReminder ? 'Edit Reminder' : 'Create New Reminder'}
              </Text>
              <TouchableOpacity onPress={onDismiss} style={styles.closeButton}>
                <Ionicons name="close" size={24} color="#666" />
              </TouchableOpacity>
            </View>

            {/* Error message */}
            {error && (
              <Card style={styles.errorCard}>
                <Card.Content>
                  <Text style={styles.errorText}>{error}</Text>
                </Card.Content>
              </Card>
            )}

            {/* Title Input */}
            <View style={styles.inputSection}>
              <Text style={styles.sectionLabel}>What would you like to be reminded about?</Text>
              <TextInput
                mode="outlined"
                value={title}
                onChangeText={setTitle}
                placeholder="e.g., Take morning medication, Doctor appointment"
                style={styles.titleInput}
                outlineColor="#E1E5E9"
                activeOutlineColor="#4285F4"
              />
            </View>

            {/* Date Selection */}
            <View style={styles.inputSection}>
              <Text style={styles.sectionLabel}>When?</Text>
              
              {/* Quick Date Presets */}
              <View style={styles.presetsContainer}>
                {DATE_PRESETS.map((preset) => (
                  <TouchableOpacity
                    key={preset.label}
                    style={[
                      styles.presetChip,
                      selectedDatePreset === preset.label && styles.selectedPresetChip
                    ]}
                    onPress={() => handleDatePresetSelect(preset)}
                  >
                    <Text style={[
                      styles.presetChipText,
                      selectedDatePreset === preset.label && styles.selectedPresetChipText
                    ]}>
                      {preset.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>

              {/* Custom Date Picker */}
              <TouchableOpacity 
                style={styles.dropdownButton}
                onPress={() => setShowStartDatePicker(true)}
              >
                <View style={styles.dropdownContent}>
                  <Ionicons name="calendar-outline" size={24} color="#4285F4" />
                  <View style={styles.dropdownTextContainer}>
                    <Text style={styles.dropdownLabel}>Select Date</Text>
                    <Text style={styles.dropdownValue}>
                      {format(startDate, 'EEEE, MMMM d, yyyy')}
                    </Text>
                  </View>
                  <Ionicons name="chevron-down" size={20} color="#666" />
                </View>
              </TouchableOpacity>
            </View>

            {/* Time Selection */}
            <View style={styles.inputSection}>
              <Text style={styles.sectionLabel}>What time?</Text>
              
              {/* Quick Time Presets */}
              <View style={styles.presetsContainer}>
                {TIME_PRESETS.map((preset) => (
                  <TouchableOpacity
                    key={preset.label}
                    style={[
                      styles.presetChip,
                      selectedTimePreset === preset.label && styles.selectedPresetChip
                    ]}
                    onPress={() => handleTimePresetSelect(preset)}
                  >
                    <Text style={[
                      styles.presetChipText,
                      selectedTimePreset === preset.label && styles.selectedPresetChipText
                    ]}>
                      {preset.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>

              {/* Custom Time Picker */}
              <TouchableOpacity 
                style={styles.dropdownButton}
                onPress={() => setShowStartTimePicker(true)}
              >
                <View style={styles.dropdownContent}>
                  <Ionicons name="time-outline" size={24} color="#4285F4" />
                  <View style={styles.dropdownTextContainer}>
                    <Text style={styles.dropdownLabel}>Select Time</Text>
                    <Text style={styles.dropdownValue}>
                      {format(startDate, 'h:mm a')}
                    </Text>
                  </View>
                  <Ionicons name="chevron-down" size={20} color="#666" />
                </View>
              </TouchableOpacity>
            </View>

            {/* Description Input */}
            <View style={styles.inputSection}>
              <Text style={styles.sectionLabel}>Additional notes (optional)</Text>
              <TextInput
                mode="outlined"
                value={description}
                onChangeText={setDescription}
                placeholder="Add any additional details..."
                multiline
                numberOfLines={3}
                style={styles.descriptionInput}
                outlineColor="#E1E5E9"
                activeOutlineColor="#4285F4"
              />
            </View>

            {/* End Date Toggle */}
            <View style={styles.switchSection}>
              <View style={styles.switchHeader}>
                <Text style={styles.switchLabel}>Set end time</Text>
                <Switch
                  value={withEndDate}
                  onValueChange={setWithEndDate}
                  color="#4285F4"
                />
              </View>
              {withEndDate && (
                <View style={styles.endDateSection}>
                  <TouchableOpacity 
                    style={styles.dropdownButton}
                    onPress={() => setShowEndDatePicker(true)}
                  >
                    <View style={styles.dropdownContent}>
                      <Ionicons name="calendar-outline" size={24} color="#4285F4" />
                      <View style={styles.dropdownTextContainer}>
                        <Text style={styles.dropdownLabel}>End Date</Text>
                        <Text style={styles.dropdownValue}>
                          {format(endDate, 'MMM d, yyyy')}
                        </Text>
                      </View>
                      <Ionicons name="chevron-down" size={20} color="#666" />
                    </View>
                  </TouchableOpacity>
                  
                  <TouchableOpacity 
                    style={styles.dropdownButton}
                    onPress={() => setShowEndTimePicker(true)}
                  >
                    <View style={styles.dropdownContent}>
                      <Ionicons name="time-outline" size={24} color="#4285F4" />
                      <View style={styles.dropdownTextContainer}>
                        <Text style={styles.dropdownLabel}>End Time</Text>
                        <Text style={styles.dropdownValue}>
                          {format(endDate, 'h:mm a')}
                        </Text>
                      </View>
                      <Ionicons name="chevron-down" size={20} color="#666" />
                    </View>
                  </TouchableOpacity>
                </View>
              )}
            </View>

            {/* Recurring Reminder Toggle */}
            <View style={styles.switchSection}>
              <View style={styles.switchHeader}>
                <Text style={styles.switchLabel}>Repeat reminder</Text>
                <Switch
                  value={recurring}
                  onValueChange={(value) => {
                    setRecurring(value);
                    if (value) setShowRecurrenceOptions(true);
                    else setShowRecurrenceOptions(false);
                  }}
                  color="#4285F4"
                />
              </View>
              
              {recurring && (
                <View style={styles.recurrenceSection}>
                  <Text style={styles.recurrenceLabel}>Repeat frequency:</Text>
                  <RadioButton.Group
                    onValueChange={setRecurrence}
                    value={recurrence || 'daily'}
                  >
                    {[
                      { value: 'daily', label: 'Daily' },
                      { value: 'weekly', label: 'Weekly' },
                      { value: 'monthly', label: 'Monthly' }
                    ].map((option) => (
                      <TouchableOpacity
                        key={option.value}
                        style={styles.radioOption}
                        onPress={() => setRecurrence(option.value)}
                      >
                        <RadioButton value={option.value} color="#4285F4" />
                        <Text style={styles.radioLabel}>{option.label}</Text>
                      </TouchableOpacity>
                    ))}
                  </RadioButton.Group>
                </View>
              )}
            </View>

            {/* Action Buttons */}
            <View style={styles.actionButtons}>
              <Button
                mode="outlined"
                onPress={onDismiss}
                style={styles.cancelButton}
                labelStyle={styles.cancelButtonText}
                disabled={loading}
              >
                Cancel
              </Button>
              <Button
                mode="contained"
                onPress={handleSubmit}
                style={styles.saveButton}
                labelStyle={styles.saveButtonText}
                loading={loading}
                disabled={loading || !title.trim()}
              >
                {editReminder ? 'Update' : 'Create'} Reminder
              </Button>
            </View>
          </ScrollView>

          {/* Date/Time Pickers */}
          {showStartDatePicker && (
            <Modal
              visible={showStartDatePicker}
              onDismiss={() => setShowStartDatePicker(false)}
              contentContainerStyle={styles.datePickerModal}
            >
              <View style={styles.datePickerContainer}>
                <View style={styles.datePickerHeader}>
                  <Text style={styles.datePickerTitle}>Select Date</Text>
                  <TouchableOpacity 
                    onPress={() => setShowStartDatePicker(false)}
                    style={styles.closeDatePicker}
                  >
                    <Ionicons name="close" size={24} color="#666" />
                  </TouchableOpacity>
                </View>
                
                {Platform.OS === 'web' ? (
                  <input
                    type="date"
                    value={format(startDate, 'yyyy-MM-dd')}
                    onChange={(e) => {
                      if (e.target.value) {
                        const newDate = new Date(startDate);
                        const selectedDate = new Date(e.target.value);
                        newDate.setFullYear(selectedDate.getFullYear());
                        newDate.setMonth(selectedDate.getMonth());
                        newDate.setDate(selectedDate.getDate());
                        setStartDate(newDate);
                        setSelectedDatePreset(null);
                        
                        // Update end date to be at least 1 hour after start date
                        const minEndDate = new Date(newDate.getTime() + 60 * 60 * 1000);
                        if (endDate < minEndDate) {
                          setEndDate(minEndDate);
                        }
                      }
                      setShowStartDatePicker(false);
                    }}
                    style={{
                      width: '100%',
                      padding: 16,
                      fontSize: 16,
                      borderRadius: 8,
                      border: '1px solid #E2E8F0',
                      backgroundColor: '#F8FAFC'
                    }}
                  />
                ) : (
                  <TextInput
                    value={format(startDate, 'yyyy-MM-dd')}
                    onChangeText={(text) => {
                      const newDate = new Date(text);
                      if (!isNaN(newDate.getTime())) {
                        setStartDate(newDate);
                        setSelectedDatePreset(null);
                      }
                      setShowStartDatePicker(false);
                    }}
                    placeholder="YYYY-MM-DD"
                    style={styles.datePickerInput}
                  />
                )}
              </View>
            </Modal>
          )}
          
          {showStartTimePicker && (
            <Modal
              visible={showStartTimePicker}
              onDismiss={() => setShowStartTimePicker(false)}
              contentContainerStyle={styles.datePickerModal}
            >
              <View style={styles.datePickerContainer}>
                <View style={styles.datePickerHeader}>
                  <Text style={styles.datePickerTitle}>Select Time</Text>
                  <TouchableOpacity 
                    onPress={() => setShowStartTimePicker(false)}
                    style={styles.closeDatePicker}
                  >
                    <Ionicons name="close" size={24} color="#666" />
                  </TouchableOpacity>
                </View>
                
                {Platform.OS === 'web' ? (
                  <input
                    type="time"
                    value={format(startDate, 'HH:mm')}
                    onChange={(e) => {
                      if (e.target.value) {
                        const [hours, minutes] = e.target.value.split(':');
                        const newDate = new Date(startDate);
                        newDate.setHours(parseInt(hours));
                        newDate.setMinutes(parseInt(minutes));
                        setStartDate(newDate);
                        setSelectedTimePreset(null);
                        
                        // Update end date to be at least 1 hour after start date
                        const minEndDate = new Date(newDate.getTime() + 60 * 60 * 1000);
                        if (endDate < minEndDate) {
                          setEndDate(minEndDate);
                        }
                      }
                      setShowStartTimePicker(false);
                    }}
                    style={{
                      width: '100%',
                      padding: 16,
                      fontSize: 16,
                      borderRadius: 8,
                      border: '1px solid #E2E8F0',
                      backgroundColor: '#F8FAFC'
                    }}
                  />
                ) : (
                  <TextInput
                    value={format(startDate, 'HH:mm')}
                    onChangeText={(text) => {
                      const [hours, minutes] = text.split(':');
                      if (hours && minutes) {
                        const newDate = new Date(startDate);
                        newDate.setHours(parseInt(hours));
                        newDate.setMinutes(parseInt(minutes));
                        setStartDate(newDate);
                        setSelectedTimePreset(null);
                      }
                      setShowStartTimePicker(false);
                    }}
                    placeholder="HH:MM"
                    style={styles.datePickerInput}
                  />
                )}
              </View>
            </Modal>
          )}
          
          {showEndDatePicker && withEndDate && (
            <Modal
              visible={showEndDatePicker}
              onDismiss={() => setShowEndDatePicker(false)}
              contentContainerStyle={styles.datePickerModal}
            >
              <View style={styles.datePickerContainer}>
                <View style={styles.datePickerHeader}>
                  <Text style={styles.datePickerTitle}>Select End Date</Text>
                  <TouchableOpacity 
                    onPress={() => setShowEndDatePicker(false)}
                    style={styles.closeDatePicker}
                  >
                    <Ionicons name="close" size={24} color="#666" />
                  </TouchableOpacity>
                </View>
                
                {Platform.OS === 'web' ? (
                  <input
                    type="date"
                    value={format(endDate, 'yyyy-MM-dd')}
                    onChange={(e) => {
                      if (e.target.value) {
                        const newDate = new Date(endDate);
                        const selectedDate = new Date(e.target.value);
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
                      setShowEndDatePicker(false);
                    }}
                    style={{
                      width: '100%',
                      padding: 16,
                      fontSize: 16,
                      borderRadius: 8,
                      border: '1px solid #E2E8F0',
                      backgroundColor: '#F8FAFC'
                    }}
                  />
                ) : (
                  <TextInput
                    value={format(endDate, 'yyyy-MM-dd')}
                    onChangeText={(text) => {
                      const newDate = new Date(text);
                      if (!isNaN(newDate.getTime()) && newDate >= startDate) {
                        setEndDate(newDate);
                      }
                      setShowEndDatePicker(false);
                    }}
                    placeholder="YYYY-MM-DD"
                    style={styles.datePickerInput}
                  />
                )}
              </View>
            </Modal>
          )}
          
          {showEndTimePicker && withEndDate && (
            <Modal
              visible={showEndTimePicker}
              onDismiss={() => setShowEndTimePicker(false)}
              contentContainerStyle={styles.datePickerModal}
            >
              <View style={styles.datePickerContainer}>
                <View style={styles.datePickerHeader}>
                  <Text style={styles.datePickerTitle}>Select End Time</Text>
                  <TouchableOpacity 
                    onPress={() => setShowEndTimePicker(false)}
                    style={styles.closeDatePicker}
                  >
                    <Ionicons name="close" size={24} color="#666" />
                  </TouchableOpacity>
                </View>
                
                {Platform.OS === 'web' ? (
                  <input
                    type="time"
                    value={format(endDate, 'HH:mm')}
                    onChange={(e) => {
                      if (e.target.value) {
                        const [hours, minutes] = e.target.value.split(':');
                        const newDate = new Date(endDate);
                        newDate.setHours(parseInt(hours));
                        newDate.setMinutes(parseInt(minutes));
                        
                        // Ensure end time is not before start time on the same day
                        if (newDate.toDateString() === startDate.toDateString() && newDate <= startDate) {
                          setEndDate(new Date(startDate.getTime() + 60 * 60 * 1000));
                        } else {
                          setEndDate(newDate);
                        }
                      }
                      setShowEndTimePicker(false);
                    }}
                    style={{
                      width: '100%',
                      padding: 16,
                      fontSize: 16,
                      borderRadius: 8,
                      border: '1px solid #E2E8F0',
                      backgroundColor: '#F8FAFC'
                    }}
                  />
                ) : (
                  <TextInput
                    value={format(endDate, 'HH:mm')}
                    onChangeText={(text) => {
                      const [hours, minutes] = text.split(':');
                      if (hours && minutes) {
                        const newDate = new Date(endDate);
                        newDate.setHours(parseInt(hours));
                        newDate.setMinutes(parseInt(minutes));
                        setEndDate(newDate);
                      }
                      setShowEndTimePicker(false);
                    }}
                    placeholder="HH:MM"
                    style={styles.datePickerInput}
                  />
                )}
              </View>
            </Modal>
          )}
        </KeyboardAvoidingView>
      </Modal>
    </Portal>
  );
};

const styles = StyleSheet.create({
  modalContainer: {
    flex: 1,
    backgroundColor: 'white',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    marginTop: 60,
    overflow: 'hidden',
  },
  keyboardAvoidingView: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 24,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
    paddingHorizontal: 4,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1F2937',
    flex: 1,
  },
  closeButton: {
    backgroundColor: '#F3F4F6',
    borderRadius: 20,
    padding: 8,
    marginLeft: 12,
  },
  errorCard: {
    backgroundColor: '#FEF2F2',
    borderRadius: 12,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#FECACA',
  },
  errorText: {
    color: '#DC2626',
    fontSize: 14,
    fontWeight: '500',
  },
  inputSection: {
    marginBottom: 20,
  },
  sectionLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 12,
  },
  titleInput: {
    backgroundColor: '#F8FAFC',
    borderRadius: 12,
    fontSize: 16,
  },
  presetsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 16,
  },
  presetChip: {
    backgroundColor: '#F8FAFC',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  selectedPresetChip: {
    backgroundColor: '#EEF2FF',
    borderColor: '#4285F4',
    borderWidth: 2,
  },
  presetChipText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
  },
  selectedPresetChipText: {
    color: '#4285F4',
    fontWeight: '600',
  },
  dropdownButton: {
    backgroundColor: '#F8FAFC',
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#E2E8F0',
    marginTop: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  dropdownContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  dropdownTextContainer: {
    flex: 1,
    marginLeft: 12,
  },
  dropdownLabel: {
    fontSize: 12,
    fontWeight: '500',
    color: '#64748B',
    marginBottom: 2,
  },
  dropdownValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
  switchSection: {
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
  switchHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  switchLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
  endDateSection: {
    gap: 12,
    marginTop: 16,
  },
  recurrenceSection: {
    marginTop: 16,
  },
  recurrenceLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 8,
  },
  radioOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: '#F8FAFC',
    marginVertical: 4,
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  radioLabel: {
    fontSize: 16,
    fontWeight: '500',
    color: '#374151',
    marginLeft: 12,
    flex: 1,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
    paddingVertical: 20,
    paddingBottom: 40,
  },
  cancelButton: {
    flex: 1,
    borderRadius: 12,
    borderColor: '#E2E8F0',
    borderWidth: 1,
  },
  cancelButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#64748B',
  },
  saveButton: {
    flex: 1,
    borderRadius: 12,
    backgroundColor: '#4285F4',
  },
  saveButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: 'white',
  },
  descriptionInput: {
    backgroundColor: '#F8FAFC',
    borderRadius: 12,
    fontSize: 16,
    minHeight: 80,
    textAlignVertical: 'top',
  },
  datePickerModal: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  datePickerContainer: {
    flex: 1,
    backgroundColor: 'white',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    marginTop: 60,
    overflow: 'hidden',
  },
  datePickerHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
  },
  datePickerTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1F2937',
  },
  closeDatePicker: {
    backgroundColor: '#F3F4F6',
    borderRadius: 20,
    padding: 8,
  },
  datePickerInput: {
    flex: 1,
    padding: 16,
  },
});

export default ReminderForm; 