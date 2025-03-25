import React, { useState, useEffect, useCallback } from 'react';
import { StyleSheet, View, ScrollView, TouchableOpacity, FlatList, Alert, Modal, TextInput } from 'react-native';
import { Text, Button, Card, FAB, Chip, Avatar, IconButton, Divider, Menu, Portal, ActivityIndicator } from 'react-native-paper';
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from '@expo/vector-icons';
import { Calendar } from 'react-native-calendars';
import { useFocusEffect } from '@react-navigation/native';
import ReminderForm from '../components/ReminderForm';
import { reminderService, ReminderData, ReminderType, googleCalendarService } from '../services/reminderService';

type ReminderCategory = 'medication' | 'appointment' | 'activity' | 'hydration';

// Define our own MarkedDates interface
interface MarkedDateItem {
  selected?: boolean;
  marked?: boolean;
  selectedColor?: string;
  dotColor?: string;
}

interface MarkedDates {
  [date: string]: MarkedDateItem;
}

interface RemindersScreenProps {
  setActiveTab: (tab: string) => void;
}

export default function RemindersScreen({ setActiveTab }: RemindersScreenProps) {
  // State for date selection and calendar
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [markedDates, setMarkedDates] = useState<MarkedDates>({
    [selectedDate]: { selected: true, selectedColor: '#4285F4' },
  });
  
  // UI state
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filterVisible, setFilterVisible] = useState(false);
  const [activeFilter, setActiveFilter] = useState<ReminderCategory | 'all'>('all');
  const [menuVisible, setMenuVisible] = useState(false);
  const [voiceModalVisible, setVoiceModalVisible] = useState(false);
  const [voiceInput, setVoiceInput] = useState('');
  const [processingVoice, setProcessingVoice] = useState(false);
  const [aiResponse, setAiResponse] = useState('');
  
  // Form state
  const [formVisible, setFormVisible] = useState(false);
  const [editingReminder, setEditingReminder] = useState<ReminderData | undefined>(undefined);
  
  // Google Calendar state
  const [isGoogleConnected, setIsGoogleConnected] = useState(false);
  const [syncingWithGoogle, setSyncingWithGoogle] = useState(false);
  
  // Data state
  const [reminders, setReminders] = useState<ReminderType[]>([]);
  
  // Check Google Calendar connection
  const checkGoogleConnection = useCallback(async () => {
    try {
      console.log('Checking Google Calendar connection...');
      const connected = await googleCalendarService.checkConnection();
      console.log('Google Calendar connection status:', connected);
      setIsGoogleConnected(connected);
    } catch (error: any) {
      console.error('Error checking Google connection:', error);
      const errorMessage = error.response?.data?.message || error.message || 'Unknown error';
      Alert.alert('Connection Error', `Failed to check Google Calendar status: ${errorMessage}`);
    }
  }, []);
  
  // Fetch reminders for the selected date
  const fetchReminders = useCallback(async () => {
    try {
      console.log('Fetching reminders for date:', selectedDate);
      setLoading(true);
      const data = await reminderService.getRemindersForDate(selectedDate);
      console.log('Fetched reminders:', data.length);
      setReminders(data);
      
      // Update calendar markers based on all reminders
      updateCalendarMarkers();
    } catch (error: any) {
      console.error('Error fetching reminders:', error);
      const errorMessage = error.response?.data?.message || error.message || 'Unknown error';
      Alert.alert('Data Error', `Failed to load reminders: ${errorMessage}`);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [selectedDate]);
  
  // Update calendar markers
  const updateCalendarMarkers = useCallback(async () => {
    try {
      // Get all reminders to mark dates in calendar
      const allReminders = await reminderService.getAllReminders();
      
      // Group reminders by date
      const dateMap: Record<string, ReminderType[]> = {};
      allReminders.forEach(reminder => {
        if (!dateMap[reminder.date]) {
          dateMap[reminder.date] = [];
        }
        dateMap[reminder.date].push(reminder);
      });
      
      // Create marked dates for calendar
      const newMarkedDates: MarkedDates = {
        [selectedDate]: { selected: true, selectedColor: '#4285F4' }
      };
      
      // Add dots for dates with reminders
      Object.keys(dateMap).forEach(date => {
        const remindersForDate = dateMap[date];
        
        // Skip currently selected date (we already marked it as selected)
        if (date === selectedDate) {
          newMarkedDates[date] = {
            ...newMarkedDates[date],
            marked: true,
            dotColor: '#4285F4'
          };
          return;
        }
        
        // For other dates, just mark them
        newMarkedDates[date] = {
          marked: true,
          dotColor: '#4285F4'
        };
      });
      
      setMarkedDates(newMarkedDates);
    } catch (error) {
      console.error('Error updating calendar markers:', error);
    }
  }, [selectedDate]);
  
  // Sync reminders with Google Calendar
  const syncWithGoogleCalendar = async () => {
    try {
      setSyncingWithGoogle(true);
      const result = await googleCalendarService.syncReminders();
      
      if (result.success) {
        Alert.alert(
          'Sync Complete',
          result.message,
          [{ text: 'OK', onPress: () => fetchReminders() }]
        );
      } else {
        Alert.alert('Sync Failed', result.message);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to sync with Google Calendar');
    } finally {
      setSyncingWithGoogle(false);
    }
  };
  
  // Import events from Google Calendar
  const importFromGoogleCalendar = async () => {
    try {
      setSyncingWithGoogle(true);
      const result = await googleCalendarService.importEvents(30);
      
      if (result.success) {
        Alert.alert(
          'Import Complete',
          result.message,
          [{ text: 'OK', onPress: () => {
            fetchReminders();
            updateCalendarMarkers();
          }}]
        );
      } else {
        Alert.alert('Import Failed', result.message);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to import events from Google Calendar');
    } finally {
      setSyncingWithGoogle(false);
    }
  };
  
  // Connect to Google Calendar
  const connectToGoogleCalendar = async () => {
    try {
      const authUrl = await googleCalendarService.getAuthURL();
      if (authUrl) {
        // Open Google auth URL in a browser
        Alert.alert(
          'Connect to Google Calendar',
          'You will be redirected to Google to authorize access to your calendar. After authorization, please return to the app.',
          [
            { text: 'Cancel', style: 'cancel' },
            { 
              text: 'Continue', 
              onPress: () => {
                // Use Linking.openURL in a real app
                Alert.alert('Open URL', `Would open: ${authUrl}`);
                // This would actually be implemented with Linking from react-native
                // Linking.openURL(authUrl);
              } 
            }
          ]
        );
      } else {
        Alert.alert('Error', 'Failed to get authorization URL');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to connect to Google Calendar');
    }
  };
  
  // Disconnect from Google Calendar
  const disconnectFromGoogleCalendar = () => {
    Alert.alert(
      'Disconnect Google Calendar',
      'Are you sure you want to disconnect your Google Calendar? Your reminders will no longer sync.',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Disconnect', 
          style: 'destructive',
          onPress: async () => {
            try {
              const success = await googleCalendarService.disconnect();
              if (success) {
                setIsGoogleConnected(false);
                Alert.alert('Success', 'Your Google Calendar has been disconnected');
              } else {
                Alert.alert('Error', 'Failed to disconnect your Google Calendar');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to disconnect from Google Calendar');
            }
          }
        }
      ]
    );
  };
  
  // Handle date selection in calendar
  const handleDateSelect = (day: any) => {
    const selectedDay = day.dateString;
    
    // Update marked dates
    const updatedMarkedDates: MarkedDates = { ...markedDates };
    
    // Remove previous selection
    if (markedDates[selectedDate]) {
      const { selected, selectedColor, ...rest } = markedDates[selectedDate];
      updatedMarkedDates[selectedDate] = rest;
      
      // If there are no other properties, remove the date
      if (Object.keys(updatedMarkedDates[selectedDate]).length === 0) {
        delete updatedMarkedDates[selectedDate];
      }
    }
    
    // Add new selection
    updatedMarkedDates[selectedDay] = { 
      ...(updatedMarkedDates[selectedDay] || {}),
      selected: true, 
      selectedColor: '#4285F4' 
    };
    
    setMarkedDates(updatedMarkedDates);
    setSelectedDate(selectedDay);
  };
  
  // Toggle reminder completion
  const toggleReminderCompletion = async (id: string, completed: boolean) => {
    try {
      await reminderService.toggleCompletion(id, !completed);
      
      // Update local state
      setReminders(prevReminders => 
        prevReminders.map(reminder => 
          reminder.id === id 
            ? { ...reminder, completed: !completed } 
            : reminder
        )
      );
    } catch (error) {
      console.error('Error toggling completion:', error);
      Alert.alert('Error', 'Failed to update reminder status');
    }
  };
  
  // Delete reminder
  const handleDeleteReminder = (id: string) => {
    Alert.alert(
      'Delete Reminder',
      'Are you sure you want to delete this reminder?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Delete', 
          style: 'destructive',
          onPress: async () => {
            try {
              await reminderService.deleteReminder(id);
              
              // Update local state
              setReminders(prevReminders => 
                prevReminders.filter(reminder => reminder.id !== id)
              );
            } catch (error) {
              console.error('Error deleting reminder:', error);
              Alert.alert('Error', 'Failed to delete reminder');
            }
          }
        }
      ]
    );
  };
  
  // Edit reminder
  const handleEditReminder = (reminder: ReminderType) => {
    // Convert UI reminder to API format
    const apiReminder: ReminderData = {
      _id: reminder.id,
      title: reminder.title,
      description: reminder.description,
      start_time: new Date(`${reminder.date}T${reminder.time}`).toISOString(),
      recurrence: reminder.recurring ? (reminder.recurrencePattern?.toLowerCase() as any) : null,
      completed: reminder.completed,
      google_event_id: reminder.google_event_id
    };
    
    setEditingReminder(apiReminder);
    setFormVisible(true);
  };
  
  // Process voice reminder input
  const processVoiceReminder = async () => {
    if (!voiceInput.trim()) {
      setAiResponse('Please enter a reminder text');
      return;
    }

    setProcessingVoice(true);
    setAiResponse('Processing your request...');

    try {
      const result = await reminderService.setVoiceReminder(voiceInput);

      if (result.success) {
        setAiResponse(result.message);
        
        // Refresh reminders list
        setTimeout(() => {
          fetchReminders();
        }, 1000);
        
        // Reset input after successful processing
        setTimeout(() => {
          setVoiceInput('');
          setProcessingVoice(false);
          setVoiceModalVisible(false);
        }, 3000);
      } else {
        setAiResponse(result.message || 'Failed to process reminder');
        setProcessingVoice(false);
      }
    } catch (error) {
      console.error('Error processing voice reminder:', error);
      setAiResponse('Error connecting to the server');
      setProcessingVoice(false);
    }
  };
  
  // Initialize
  useEffect(() => {
    checkGoogleConnection();
    fetchReminders();
  }, [fetchReminders, checkGoogleConnection]);
  
  // Refresh data when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      checkGoogleConnection();
      fetchReminders();
      return () => {};
    }, [fetchReminders, checkGoogleConnection])
  );
  
  // Filter reminders based on active filter
  const filteredReminders = activeFilter === 'all' 
    ? reminders 
    : reminders.filter(reminder => reminder.type === activeFilter);
  
  // Get reminders by type
  const getRemindersCountByType = (type: ReminderCategory): number => {
    return reminders.filter(r => r.type === type).length;
  };
  
  // Get completion statistics
  const getCompletionRate = (): number => {
    if (reminders.length === 0) return 0;
    const completedCount = reminders.filter(r => r.completed).length;
    return Math.round((completedCount / reminders.length) * 100);
  };
  
  // Render reminder item
  const renderReminderItem = ({ item }: { item: ReminderType }) => (
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
            onPress={() => toggleReminderCompletion(item.id, item.completed)}>
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
        
        {item.google_event_id && (
          <View style={styles.recurrenceContainer}>
            <Ionicons name="logo-google" size={16} color="#777" />
            <Text style={styles.recurrenceText}>Synced with Google Calendar</Text>
          </View>
        )}
        
        <View style={styles.reminderActions}>
          <Button 
            mode="text" 
            compact 
            onPress={() => handleEditReminder(item)}
            style={styles.actionButton}
            labelStyle={styles.actionButtonLabel}>
            Edit
          </Button>
          <Button 
            mode="text" 
            compact 
            onPress={() => handleDeleteReminder(item.id)}
            style={styles.actionButton}
            labelStyle={[styles.actionButtonLabel, {color: '#FF3B30'}]}>
            Delete
          </Button>
        </View>
      </Card.Content>
    </Card>
  );
  
  // Render voice reminder modal
  const renderVoiceReminderModal = () => {
    return (
      <Portal>
        <Modal
          visible={voiceModalVisible}
          onDismiss={() => setVoiceModalVisible(false)}
          style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Set a Reminder with Voice</Text>
            <Text style={styles.modalSubtitle}>
              Type your reminder as you would say it, for example:
            </Text>
            <Text style={styles.exampleText}>
              "Remind me to take my medicine at 3 pm"
            </Text>
            
            <TextInput
              style={styles.voiceInput}
              placeholder="Enter your reminder here..."
              value={voiceInput}
              onChangeText={setVoiceInput}
              multiline
            />
            
            {aiResponse ? (
              <View style={styles.responseContainer}>
                <Text style={styles.responseText}>{aiResponse}</Text>
              </View>
            ) : null}
            
            <View style={styles.modalActions}>
              <Button
                mode="outlined"
                onPress={() => setVoiceModalVisible(false)}
                style={styles.modalButton}>
                Cancel
              </Button>
              <Button
                mode="contained"
                onPress={processVoiceReminder}
                style={styles.modalButton}
                loading={processingVoice}
                disabled={processingVoice || !voiceInput.trim()}>
                Process
              </Button>
            </View>
          </View>
        </Modal>
      </Portal>
    );
  };
  
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
          {isGoogleConnected ? (
            <>
              <Menu.Item 
                onPress={syncWithGoogleCalendar} 
                title="Sync with Google Calendar" 
                leadingIcon="sync"
                disabled={syncingWithGoogle}
              />
              <Menu.Item 
                onPress={disconnectFromGoogleCalendar}
                title="Disconnect Google Calendar" 
                leadingIcon="google"
              />
            </>
          ) : (
            <Menu.Item 
              onPress={connectToGoogleCalendar}
              title="Connect Google Calendar" 
              leadingIcon="google"
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
              <Text style={styles.calendarTitle}>Calendar</Text>
              {isGoogleConnected && (
                <Chip 
                  icon="google" 
                  mode="outlined" 
                  style={styles.syncChip}>
                  Google Synced
                </Chip>
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
              All ({reminders.length})
            </Chip>
            <Chip 
              selected={activeFilter === 'medication'}
              onPress={() => setActiveFilter('medication')}
              style={styles.filterChip}
              selectedColor="#FF9500">
              Medications ({getRemindersCountByType('medication')})
            </Chip>
            <Chip 
              selected={activeFilter === 'appointment'}
              onPress={() => setActiveFilter('appointment')}
              style={styles.filterChip}
              selectedColor="#FF3B30">
              Appointments ({getRemindersCountByType('appointment')})
            </Chip>
            <Chip 
              selected={activeFilter === 'activity'}
              onPress={() => setActiveFilter('activity')}
              style={styles.filterChip}
              selectedColor="#34C759">
              Activities ({getRemindersCountByType('activity')})
            </Chip>
            <Chip 
              selected={activeFilter === 'hydration'}
              onPress={() => setActiveFilter('hydration')}
              style={styles.filterChip}
              selectedColor="#4285F4">
              Hydration ({getRemindersCountByType('hydration')})
            </Chip>
          </ScrollView>
        </View>

        {/* Google Calendar Integration Section */}
        <Card style={styles.googleCalendarCard}>
          <Card.Content>
            <View style={styles.googleCalendarHeader}>
              <View style={styles.googleCalendarTitleContainer}>
                <Ionicons name="logo-google" size={24} color="#4285F4" style={styles.googleIcon} />
                <Text style={styles.googleCalendarTitle}>Google Calendar</Text>
              </View>
              
              {isGoogleConnected ? (
                <Chip 
                  icon="check-circle" 
                  mode="outlined" 
                  style={styles.connectedChip}>
                  Connected
                </Chip>
              ) : (
                <Chip 
                  icon="close-circle" 
                  mode="outlined" 
                  style={styles.disconnectedChip}>
                  Not Connected
                </Chip>
              )}
            </View>
            
            <Text style={styles.googleCalendarDescription}>
              {isGoogleConnected 
                ? 'Sync your reminders with Google Calendar to keep everything in one place.'
                : 'Connect your Google Calendar to sync your reminders across devices.'}
            </Text>
            
            <View style={styles.googleCalendarActions}>
              {isGoogleConnected ? (
                <>
                  <Button 
                    mode="contained" 
                    icon="sync"
                    loading={syncingWithGoogle}
                    disabled={syncingWithGoogle}
                    onPress={syncWithGoogleCalendar}
                    style={styles.syncButton}>
                    Sync Reminders
                  </Button>
                  
                  <Button 
                    mode="outlined" 
                    icon="download"
                    loading={syncingWithGoogle}
                    disabled={syncingWithGoogle}
                    onPress={importFromGoogleCalendar}
                    style={styles.importButton}>
                    Import Events
                  </Button>
                  
                  <Button 
                    mode="text" 
                    onPress={disconnectFromGoogleCalendar}
                    disabled={syncingWithGoogle}
                    style={styles.disconnectButton}
                    labelStyle={{color: '#FF3B30'}}>
                    Disconnect
                  </Button>
                </>
              ) : (
                <Button 
                  mode="contained" 
                  icon="google"
                  onPress={connectToGoogleCalendar}
                  style={styles.connectButton}>
                  Connect with Google
                </Button>
              )}
            </View>
          </Card.Content>
        </Card>

        {/* Reminders list */}
        <View style={styles.remindersSection}>
          <Text style={styles.sectionTitle}>
            Upcoming Reminders for {new Date(selectedDate).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
          </Text>
          
          {loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#4285F4" />
              <Text style={styles.loadingText}>Loading reminders...</Text>
            </View>
          ) : filteredReminders.length > 0 ? (
            <FlatList
              data={filteredReminders}
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
                  onPress={() => {
                    setEditingReminder(undefined);
                    setFormVisible(true);
                  }}
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
                  onPress={() => setVoiceModalVisible(true)}
                  icon="microphone"
                  style={styles.voiceReminderButton}>
                  Try Voice Reminder
                </Button>
              </View>
              <View style={styles.voiceReminderImageContainer}>
                <MaterialCommunityIcons name="microphone" size={72} color="#4285F4" />
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* Render the voice reminder modal */}
        {renderVoiceReminderModal()}

        {/* Reminder statistics */}
        <Card style={styles.statsCard}>
          <Card.Content>
            <Text style={styles.statsTitle}>Reminder Statistics</Text>
            <View style={styles.statsContainer}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{getCompletionRate()}%</Text>
                <Text style={styles.statLabel}>Completion Rate</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{reminders.length}</Text>
                <Text style={styles.statLabel}>Today's Reminders</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{reminders.filter(r => !r.completed).length}</Text>
                <Text style={styles.statLabel}>Pending</Text>
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* Bottom spacer */}
        <View style={styles.bottomSpacer} />
      </ScrollView>

      {/* FAB for adding new reminders */}
      <FAB
        icon="plus"
        style={styles.fab}
        onPress={() => {
          setEditingReminder(undefined);
          setFormVisible(true);
        }}
        color="white"
      />
      
      {/* Reminder Form Modal */}
      <ReminderForm
        visible={formVisible}
        onDismiss={() => setFormVisible(false)}
        onSuccess={() => {
          fetchReminders();
        }}
        editReminder={editingReminder}
      />
    </View>
  );
}

// Helper functions
const getTypeIcon = (type: ReminderCategory) => {
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

const getTypeColor = (type: ReminderCategory) => {
  switch (type) {
    case 'medication': return '#FFF5E6';
    case 'appointment': return '#FFE5E5';
    case 'activity': return '#E6F9ED';
    case 'hydration': return '#E8F1FF';
    default: return '#F5F5F5';
  }
};

// Keep the existing styles as they are
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
  statsCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 12,
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4285F4',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#777',
    textAlign: 'center',
  },
  statDivider: {
    width: 1,
    height: 40,
    backgroundColor: '#E1E1E1',
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
  modalContainer: {
    backgroundColor: 'white',
    margin: 20,
    borderRadius: 10,
    padding: 20,
    elevation: 5,
  },
  modalContent: {
    alignItems: 'stretch',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
  },
  modalSubtitle: {
    fontSize: 16,
    color: '#555',
    marginBottom: 5,
  },
  exampleText: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
    marginBottom: 15,
  },
  voiceInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 10,
    minHeight: 80,
    fontSize: 16,
    marginBottom: 15,
  },
  responseContainer: {
    backgroundColor: '#f5f5f5',
    padding: 10,
    borderRadius: 8,
    marginBottom: 15,
  },
  responseText: {
    fontSize: 14,
    color: '#444',
  },
  modalActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modalButton: {
    flex: 1,
    margin: 5,
  },
  loadingContainer: {
    padding: 20,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#777',
  },
  // Google Calendar styles
  googleCalendarCard: {
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 12,
    elevation: 3,
  },
  googleCalendarHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  googleCalendarTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  googleIcon: {
    marginRight: 8,
  },
  googleCalendarTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  connectedChip: {
    backgroundColor: '#E5F3E8',
    borderColor: '#34C759',
  },
  disconnectedChip: {
    backgroundColor: '#FFEAEA',
    borderColor: '#FF3B30',
  },
  googleCalendarDescription: {
    fontSize: 14,
    color: '#777',
    marginBottom: 16,
  },
  googleCalendarActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'flex-start',
    gap: 10,
  },
  syncButton: {
    backgroundColor: '#4285F4',
  },
  importButton: {
    borderColor: '#4285F4',
  },
  disconnectButton: {
  },
  connectButton: {
    backgroundColor: '#4285F4',
  },
}); 