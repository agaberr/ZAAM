import React, { useState, useEffect, useCallback } from 'react';
import { StyleSheet, View, ScrollView, TouchableOpacity, FlatList, Alert } from 'react-native';
import { Text, Button, Card, FAB, Chip, Avatar, IconButton, Divider, Menu, Dialog, Portal, TextInput } from 'react-native-paper';
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from '@expo/vector-icons';
import { Calendar } from 'react-native-calendars';
import DateTimePicker from '@react-native-community/datetimepicker';
import { useAuth } from '../context/AuthContext';
import { useFocusEffect } from '@react-navigation/native';

type ReminderType = 'medication' | 'appointment' | 'activity' | 'hydration';

interface Reminder {
  id: string;
  title: string;
  time: string;
  date: string;
  type: ReminderType;
  description?: string;
  location?: string;
  completed: boolean;
  recurring?: boolean;
  recurrencePattern?: string;
}

// Google Calendar Event interface
interface CalendarEvent {
  id: string;
  summary: string;
  description?: string;
  start: {
    dateTime: string;
    timeZone: string;
  };
  end: {
    dateTime: string;
    timeZone: string;
  };
  location?: string;
}

export default function RemindersScreen({ setActiveTab }) {
  const { googleCredentials, googleSignIn, googleSignOut, createCalendarEvent, getCalendarEvents, user } = useAuth();
  
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [markedDates, setMarkedDates] = useState({
    [selectedDate]: { selected: true, selectedColor: '#4285F4' },
  });
  
  const [activeFilter, setActiveFilter] = useState<ReminderType | 'all'>('all');
  const [menuVisible, setMenuVisible] = useState(false);
  const [googleEvents, setGoogleEvents] = useState<CalendarEvent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [syncStatus, setSyncStatus] = useState('Not synced');
  
  // New reminder dialog
  const [dialogVisible, setDialogVisible] = useState(false);
  const [newReminderTitle, setNewReminderTitle] = useState('');
  const [newReminderDesc, setNewReminderDesc] = useState('');
  const [newReminderDate, setNewReminderDate] = useState(new Date());
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [newReminderType, setNewReminderType] = useState<ReminderType>('appointment');

  // Fetch events when the screen is focused
  useFocusEffect(
    useCallback(() => {
      if (googleCredentials) {
        fetchCalendarEvents();
      }
    }, [googleCredentials, selectedDate])
  );
  
  // Convert Google Calendar events to app reminders
  const convertEventsToReminders = (events: CalendarEvent[]): Reminder[] => {
    return events.map(event => {
      const startDate = new Date(event.start.dateTime);
      return {
        id: event.id,
        title: event.summary,
        time: startDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        date: startDate.toISOString().split('T')[0],
        type: determineEventType(event.description || ''),
        description: event.description || '',
        location: event.location,
        completed: false,
      };
    });
  };
  
  // Determine event type based on description (could be improved with AI)
  const determineEventType = (description: string): ReminderType => {
    const desc = description.toLowerCase();
    if (desc.includes('medication') || desc.includes('pill') || desc.includes('medicine')) {
      return 'medication';
    } else if (desc.includes('doctor') || desc.includes('hospital') || desc.includes('clinic')) {
      return 'appointment';
    } else if (desc.includes('exercise') || desc.includes('activity') || desc.includes('training')) {
      return 'activity';
    } else if (desc.includes('water') || desc.includes('drink')) {
      return 'hydration';
    }
    return 'appointment'; // Default
  };
  
  // Fetch Google Calendar events
  const fetchCalendarEvents = async () => {
    if (!googleCredentials) return;
    
    try {
      setIsLoading(true);
      
      // Get start and end of the selected month
      const date = new Date(selectedDate);
      const startOfMonth = new Date(date.getFullYear(), date.getMonth(), 1);
      const endOfMonth = new Date(date.getFullYear(), date.getMonth() + 1, 0);
      
      const events = await getCalendarEvents(startOfMonth, endOfMonth);
      setGoogleEvents(events);
      
      // Update marked dates
      const newMarkedDates = { ...markedDates };
      events.forEach(event => {
        const eventDate = new Date(event.start.dateTime).toISOString().split('T')[0];
        if (newMarkedDates[eventDate]) {
          newMarkedDates[eventDate] = { 
            ...newMarkedDates[eventDate], 
            marked: true, 
            dotColor: '#4285F4' 
          };
        } else {
          newMarkedDates[eventDate] = { marked: true, dotColor: '#4285F4' };
        }
      });
      
      // Ensure selected date is still marked as selected
      if (newMarkedDates[selectedDate]) {
        newMarkedDates[selectedDate] = {
          ...newMarkedDates[selectedDate],
          selected: true,
          selectedColor: '#4285F4'
        };
      } else {
        newMarkedDates[selectedDate] = { selected: true, selectedColor: '#4285F4' };
      }
      
      setMarkedDates(newMarkedDates);
      setSyncStatus('Synced');
    } catch (error) {
      console.error('Error fetching calendar events:', error);
      Alert.alert('Error', 'Failed to fetch calendar events. Please try again.');
      setSyncStatus('Sync failed');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Google Sign In
  const handleGoogleSignIn = async () => {
    try {
      await googleSignIn();
      fetchCalendarEvents();
    } catch (error) {
      console.error('Google sign in error:', error);
      Alert.alert('Error', 'Failed to sign in with Google. Please try again.');
    }
  };
  
  // Handle Google Sign Out
  const handleGoogleSignOut = async () => {
    try {
      await googleSignOut();
      setGoogleEvents([]);
      setSyncStatus('Not synced');
    } catch (error) {
      console.error('Google sign out error:', error);
      Alert.alert('Error', 'Failed to sign out from Google. Please try again.');
    }
  };
  
  // Create a new reminder in Google Calendar
  const handleAddReminder = async () => {
    if (!newReminderTitle) {
      Alert.alert('Error', 'Please enter a title for the reminder');
      return;
    }
    
    try {
      setIsLoading(true);
      
      // Create calendar event
      await createCalendarEvent(
        newReminderTitle,
        `Type: ${newReminderType}\n${newReminderDesc}`,
        newReminderDate
      );
      
      // Reset form and close dialog
      setNewReminderTitle('');
      setNewReminderDesc('');
      setNewReminderDate(new Date());
      setDialogVisible(false);
      
      // Refresh calendar events
      await fetchCalendarEvents();
      
      Alert.alert('Success', 'Reminder added successfully');
    } catch (error) {
      console.error('Error adding reminder:', error);
      Alert.alert('Error', 'Failed to add reminder. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDateSelect = (day: any) => {
    const selectedDay = day.dateString;
    
    // Update marked dates
    const updatedMarkedDates = { ...markedDates };
    
    // Remove previous selection
    if (markedDates[selectedDate]) {
      if (markedDates[selectedDate].marked) {
        updatedMarkedDates[selectedDate] = { marked: true, dotColor: markedDates[selectedDate].dotColor };
      } else {
        delete updatedMarkedDates[selectedDate];
      }
    }
    
    // Add new selection
    if (updatedMarkedDates[selectedDay]) {
      updatedMarkedDates[selectedDay] = { 
        ...updatedMarkedDates[selectedDay], 
        selected: true, 
        selectedColor: '#4285F4' 
      };
    } else {
      updatedMarkedDates[selectedDay] = { selected: true, selectedColor: '#4285F4' };
    }
    
    setMarkedDates(updatedMarkedDates);
    setSelectedDate(selectedDay);
  };

  const handleDateChange = (event, selectedDate) => {
    setShowDatePicker(false);
    if (selectedDate) {
      setNewReminderDate(selectedDate);
    }
  };
  
  // Get reminders for selected date
  const getSelectedDateReminders = () => {
    // Convert Google Calendar events for the selected date to reminders
    const selectedDateEvents = googleEvents.filter(event => {
      const eventDate = new Date(event.start.dateTime).toISOString().split('T')[0];
      return eventDate === selectedDate;
    });
    
    const reminders = convertEventsToReminders(selectedDateEvents);
    
    // Apply filter if needed
    if (activeFilter !== 'all') {
      return reminders.filter(reminder => reminder.type === activeFilter);
    }
    
    return reminders;
  };

  const getTypeIcon = (type: ReminderType) => {
    switch (type) {
      case 'medication':
        return <MaterialCommunityIcons name="pill" size={24} color="#FF9500" />;
      case 'appointment':
        return <FontAwesome5 name="stethoscope" size={20} color="#FF3B30" />;
      case 'activity':
        return <MaterialCommunityIcons name="brain" size={24} color="#34C759" />;
      case 'hydration':
        return <Ionicons name="water" size={24} color="#4285F4" />;
      default:
        return <Ionicons name="calendar" size={24} color="#4285F4" />;
    }
  };

  const getTypeColor = (type: ReminderType) => {
    switch (type) {
      case 'medication': return '#FFF5E6';
      case 'appointment': return '#FFE5E5';
      case 'activity': return '#E6F9ED';
      case 'hydration': return '#E8F1FF';
      default: return '#F5F5F5';
    }
  };

  const toggleReminderCompletion = (id: string) => {
    // This would update the reminder's completion status in a real app
    console.log(`Toggling completion for reminder ${id}`);
  };

  const renderReminderItem = ({ item }: { item: Reminder }) => (
    <Card style={[styles.reminderCard, { backgroundColor: getTypeColor(item.type) }]}>
      <Card.Content>
        <View style={styles.reminderHeader}>
          <View style={styles.reminderTypeContainer}>
            {getTypeIcon(item.type)}
          </View>
          <View style={styles.reminderInfo}>
            <Text style={styles.reminderTitle}>{item.title}</Text>
            <Text style={styles.reminderTime}>{item.time}</Text>
          </View>
          <TouchableOpacity 
            style={[styles.completionButton, item.completed ? styles.completedButton : {}]}
            onPress={() => toggleReminderCompletion(item.id)}>
            {item.completed ? (
              <Ionicons name="checkmark" size={20} color="white" />
            ) : null}
          </TouchableOpacity>
        </View>
        
        {item.description ? (
          <Text style={styles.reminderDescription}>{item.description}</Text>
        ) : null}
        
        {item.location ? (
          <View style={styles.locationContainer}>
            <Ionicons name="location" size={16} color="#777" />
            <Text style={styles.locationText}>{item.location}</Text>
          </View>
        ) : null}
        
        {item.recurring ? (
          <View style={styles.recurrenceContainer}>
            <Ionicons name="repeat" size={16} color="#777" />
            <Text style={styles.recurrenceText}>{item.recurrencePattern}</Text>
          </View>
        ) : null}
        
        <View style={styles.reminderActions}>
          <Button 
            mode="text" 
            compact 
            onPress={() => console.log(`Edit reminder ${item.id}`)}
            style={styles.actionButton}
            labelStyle={styles.actionButtonLabel}>
            Edit
          </Button>
          <Button 
            mode="text" 
            compact 
            onPress={() => console.log(`Delete reminder ${item.id}`)}
            style={styles.actionButton}
            labelStyle={[styles.actionButtonLabel, {color: '#FF3B30'}]}>
            Delete
          </Button>
        </View>
      </Card.Content>
    </Card>
  );

  // List of reminders for the selected date
  const selectedDateReminders = getSelectedDateReminders();

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Reminders</Text>
        <Menu
          visible={menuVisible}
          onDismiss={() => setMenuVisible(false)}
          anchor={
            <IconButton
              icon="dots-vertical"
              size={24}
              onPress={() => setMenuVisible(true)}
            />
          }
          contentStyle={styles.menuContent}
        >
          {googleCredentials ? (
            <Menu.Item 
              onPress={() => {
                fetchCalendarEvents();
                setMenuVisible(false);
              }} 
              title="Sync with Google Calendar" 
              leadingIcon="sync"
            />
          ) : (
            <Menu.Item 
              onPress={() => {
                handleGoogleSignIn();
                setMenuVisible(false);
              }} 
              title="Sign in with Google" 
              leadingIcon="google"
            />
          )}
          {googleCredentials && (
            <Menu.Item 
              onPress={() => {
                handleGoogleSignOut();
                setMenuVisible(false);
              }} 
              title="Sign out from Google" 
              leadingIcon="logout"
            />
          )}
          <Divider />
          <Menu.Item 
            onPress={() => {
              console.log('Settings');
              setMenuVisible(false);
            }} 
            title="Reminder settings" 
            leadingIcon="cog"
          />
        </Menu>
      </View>

      <ScrollView style={styles.scrollView}>
        {/* Calendar section */}
        <Card style={styles.calendarCard}>
          <Card.Content>
            <View style={styles.calendarHeader}>
              <Text style={styles.calendarTitle}>Google Calendar</Text>
              {googleCredentials ? (
                <Chip 
                  icon="sync" 
                  mode="outlined" 
                  style={styles.syncChip}
                  onPress={fetchCalendarEvents}>
                  {syncStatus}
                </Chip>
              ) : (
                <Button 
                  mode="contained" 
                  icon="google" 
                  onPress={handleGoogleSignIn}
                  style={styles.googleSignInButton}>
                  Sign in
                </Button>
              )}
            </View>
            <Calendar
              markedDates={markedDates}
              onDayPress={handleDateSelect}
              theme={{
                todayTextColor: '#4285F4',
                selectedDayBackgroundColor: '#4285F4',
                dotColor: '#4285F4',
                arrowColor: '#4285F4',
              }}
              style={styles.calendar}
            />
          </Card.Content>
        </Card>

        {/* Filter chips */}
        <View style={styles.filterContainer}>
          <Text style={styles.filterTitle}>Filter by:</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filtersScrollView}>
            <Chip 
              selected={activeFilter === 'all'}
              onPress={() => setActiveFilter('all')}
              style={styles.filterChip}
              selectedColor="#4285F4">
              All
            </Chip>
            <Chip 
              selected={activeFilter === 'medication'}
              onPress={() => setActiveFilter('medication')}
              style={styles.filterChip}
              selectedColor="#FF9500">
              Medications
            </Chip>
            <Chip 
              selected={activeFilter === 'appointment'}
              onPress={() => setActiveFilter('appointment')}
              style={styles.filterChip}
              selectedColor="#FF3B30">
              Appointments
            </Chip>
            <Chip 
              selected={activeFilter === 'activity'}
              onPress={() => setActiveFilter('activity')}
              style={styles.filterChip}
              selectedColor="#34C759">
              Activities
            </Chip>
            <Chip 
              selected={activeFilter === 'hydration'}
              onPress={() => setActiveFilter('hydration')}
              style={styles.filterChip}
              selectedColor="#4285F4">
              Hydration
            </Chip>
          </ScrollView>
        </View>

        {/* Reminders list */}
        <View style={styles.remindersSection}>
          <Text style={styles.sectionTitle}>
            Reminders for {new Date(selectedDate).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
          </Text>
          
          {selectedDateReminders.length > 0 ? (
            <FlatList
              data={selectedDateReminders}
              renderItem={renderReminderItem}
              keyExtractor={item => item.id}
              scrollEnabled={false}
              contentContainerStyle={styles.remindersList}
            />
          ) : (
            <Card style={styles.emptyStateCard}>
              <Card.Content style={styles.emptyStateContent}>
                <MaterialCommunityIcons name="calendar-blank" size={50} color="#CCCCCC" />
                <Text style={styles.emptyStateText}>No reminders for this day</Text>
                <Button 
                  mode="contained" 
                  onPress={() => setDialogVisible(true)}
                  style={styles.emptyStateButton}>
                  Add Reminder
                </Button>
              </Card.Content>
            </Card>
          )}
        </View>

        {/* Voice reminders section */}
        <Card style={styles.voiceReminderCard}>
          <Card.Content>
            <View style={styles.voiceReminderContent}>
              <View style={styles.voiceReminderTextContainer}>
                <Text style={styles.voiceReminderTitle}>Set reminders with your voice</Text>
                <Text style={styles.voiceReminderDescription}>
                  Just say "Set a reminder" to your AI companion
                </Text>
                <Button 
                  mode="contained" 
                  icon="microphone" 
                  onPress={() => setActiveTab('ai')}
                  style={styles.voiceReminderButton}>
                  Try Voice Reminder
                </Button>
              </View>
              <View style={styles.voiceReminderImageContainer}>
                <MaterialCommunityIcons name="microphone-message" size={60} color="#4285F4" />
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* Bottom spacer */}
        <View style={styles.bottomSpacer} />
      </ScrollView>

      {/* Add Reminder Dialog */}
      <Portal>
        <Dialog visible={dialogVisible} onDismiss={() => setDialogVisible(false)}>
          <Dialog.Title>Add New Reminder</Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Title"
              value={newReminderTitle}
              onChangeText={setNewReminderTitle}
              mode="outlined"
              style={styles.input}
            />
            
            <TextInput
              label="Description"
              value={newReminderDesc}
              onChangeText={setNewReminderDesc}
              mode="outlined"
              multiline
              numberOfLines={3}
              style={styles.input}
            />
            
            <Text style={styles.inputLabel}>Reminder Type:</Text>
            <View style={styles.typeChipsContainer}>
              <Chip 
                selected={newReminderType === 'medication'}
                onPress={() => setNewReminderType('medication')}
                style={styles.typeChip}
                selectedColor="#FF9500">
                Medication
              </Chip>
              <Chip 
                selected={newReminderType === 'appointment'}
                onPress={() => setNewReminderType('appointment')}
                style={styles.typeChip}
                selectedColor="#FF3B30">
                Appointment
              </Chip>
              <Chip 
                selected={newReminderType === 'activity'}
                onPress={() => setNewReminderType('activity')}
                style={styles.typeChip}
                selectedColor="#34C759">
                Activity
              </Chip>
              <Chip 
                selected={newReminderType === 'hydration'}
                onPress={() => setNewReminderType('hydration')}
                style={styles.typeChip}
                selectedColor="#4285F4">
                Hydration
              </Chip>
            </View>
            
            <Text style={styles.inputLabel}>Date and Time:</Text>
            <Button 
              mode="outlined" 
              onPress={() => setShowDatePicker(true)}
              style={styles.datePickerButton}>
              {newReminderDate.toLocaleString()}
            </Button>
            
            {showDatePicker && (
              <DateTimePicker
                value={newReminderDate}
                mode="datetime"
                is24Hour={false}
                onChange={handleDateChange}
              />
            )}
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setDialogVisible(false)}>Cancel</Button>
            <Button onPress={handleAddReminder} loading={isLoading}>Add</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>

      {/* FAB for adding new reminders */}
      <FAB
        icon="plus"
        style={styles.fab}
        onPress={() => setDialogVisible(true)}
        color="white"
        loading={isLoading}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 10,
    paddingVertical: 10,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#E1E1E1',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  menuContent: {
    marginTop: 40,
  },
  scrollView: {
    flex: 1,
  },
  calendarCard: {
    margin: 16,
    borderRadius: 12,
    elevation: 2,
  },
  calendarHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  calendarTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  syncChip: {
    backgroundColor: '#E8F1FF',
  },
  googleSignInButton: {
    backgroundColor: '#4285F4',
  },
  calendar: {
    borderRadius: 10,
    elevation: 0,
  },
  filterContainer: {
    marginHorizontal: 16,
    marginBottom: 16,
  },
  filterTitle: {
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 8,
  },
  filtersScrollView: {
    flexDirection: 'row',
  },
  filterChip: {
    marginRight: 8,
  },
  remindersSection: {
    marginHorizontal: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  remindersList: {
    gap: 12,
  },
  reminderCard: {
    borderRadius: 12,
    elevation: 1,
  },
  reminderHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  reminderTypeContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  reminderInfo: {
    flex: 1,
  },
  reminderTitle: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  reminderTime: {
    fontSize: 14,
    color: '#777',
  },
  completionButton: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#4285F4',
    justifyContent: 'center',
    alignItems: 'center',
  },
  completedButton: {
    backgroundColor: '#4285F4',
  },
  reminderDescription: {
    fontSize: 14,
    marginBottom: 8,
    paddingLeft: 52,
  },
  locationContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
    paddingLeft: 52,
  },
  locationText: {
    fontSize: 14,
    color: '#777',
    marginLeft: 4,
  },
  recurrenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    paddingLeft: 52,
  },
  recurrenceText: {
    fontSize: 14,
    color: '#777',
    marginLeft: 4,
  },
  reminderActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  actionButton: {
    marginLeft: 8,
  },
  actionButtonLabel: {
    fontSize: 14,
    color: '#4285F4',
  },
  emptyStateCard: {
    borderRadius: 12,
    marginBottom: 16,
  },
  emptyStateContent: {
    alignItems: 'center',
    padding: 20,
  },
  emptyStateText: {
    fontSize: 16,
    color: '#777',
    marginVertical: 10,
  },
  emptyStateButton: {
    marginTop: 10,
    backgroundColor: '#4285F4',
  },
  voiceReminderCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 12,
    backgroundColor: '#E8F1FF',
  },
  voiceReminderContent: {
    flexDirection: 'row',
  },
  voiceReminderTextContainer: {
    flex: 2,
  },
  voiceReminderTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  voiceReminderDescription: {
    fontSize: 14,
    marginBottom: 12,
  },
  voiceReminderButton: {
    backgroundColor: '#4285F4',
    borderRadius: 20,
    width: 180,
  },
  voiceReminderImageContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  bottomSpacer: {
    height: 80,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 80,
    backgroundColor: '#4285F4',
  },
  input: {
    marginBottom: 12,
  },
  inputLabel: {
    fontSize: 16,
    marginTop: 8,
    marginBottom: 8,
  },
  typeChipsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 12,
  },
  typeChip: {
    margin: 4,
  },
  datePickerButton: {
    marginBottom: 12,
  },
}); 