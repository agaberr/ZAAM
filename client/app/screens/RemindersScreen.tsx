import React, { useState, useEffect, useCallback } from 'react';
import { StyleSheet, View, ScrollView, TouchableOpacity, FlatList, Alert, Modal, TextInput } from 'react-native';
import { Text, Button, Card, FAB, Chip, Avatar, IconButton, Divider, Menu, Portal, ActivityIndicator } from 'react-native-paper';
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from '@expo/vector-icons';
import { Calendar } from 'react-native-calendars';
import { useFocusEffect } from 'expo-router';
import ReminderForm from '../components/ReminderForm';
import { reminderService, ReminderData, ReminderType } from '../services/reminderService';
import { format, parseISO } from 'date-fns';

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
  const [selectedDate, setSelectedDate] = useState(format(new Date(), 'yyyy-MM-dd'));
  const [markedDates, setMarkedDates] = useState<MarkedDates>({
    [format(new Date(), 'yyyy-MM-dd')]: { selected: true, selectedColor: '#4285F4' },
  });
  
  // UI state
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filterVisible, setFilterVisible] = useState(false);
  const [activeFilter, setActiveFilter] = useState<ReminderCategory | 'all'>('all');
  const [menuVisible, setMenuVisible] = useState(false);
  const [voiceModalVisible, setVoiceModalVisible] = useState(false);
  const [voicePrompt, setVoicePrompt] = useState('');
  const [processing, setProcessing] = useState(false);
  const [aiResponse, setAiResponse] = useState('');
  
  // Form state
  const [formVisible, setFormVisible] = useState(false);
  const [editingReminder, setEditingReminder] = useState<ReminderData | undefined>(undefined);
  
  // Data state
  const [reminders, setReminders] = useState<ReminderType[]>([]);
  
  // Fetch reminders for the selected date
  const fetchReminders = useCallback(async () => {
    try {
      console.log('Fetching reminders for date:', selectedDate);
      setLoading(true);
      const data = await reminderService.getRemindersForDate(selectedDate);
      
      console.log('Fetched reminders:', data);
      setReminders(data);
      
      // Update marked dates with the new data
      updateCalendarMarkers();
    } catch (error) {
      console.error('Error fetching reminders:', error);
      Alert.alert('Error', 'Failed to load reminders');
    } finally {
      setLoading(false);
    }
  }, [selectedDate]);
  
  // Update calendar markers
  const updateCalendarMarkers = () => {
    const newMarkedDates: MarkedDates = {
      [selectedDate]: { selected: true, selectedColor: '#4285F4' }
    };
    
    // Mark dates that have reminders
    reminders.forEach(reminder => {
      const date = reminder.date;
      if (date !== selectedDate) {
        newMarkedDates[date] = { 
          ...newMarkedDates[date],
          marked: true, 
          dotColor: '#4285F4' 
        };
      }
    });
    
    setMarkedDates(newMarkedDates);
  };
  
  // Handle date selection in calendar
  const handleDateSelect = (day: any) => {
    const selectedDay = day.dateString;
    
    // Update selected date in marked dates
    const updatedMarkedDates = {
      ...markedDates,
    };
    
    // Remove selection from previous date
    if (markedDates[selectedDate]) {
      const { selected, selectedColor, ...rest } = markedDates[selectedDate];
      updatedMarkedDates[selectedDate] = rest;
    }
    
    // Add selection to new date
    updatedMarkedDates[selectedDay] = {
      ...updatedMarkedDates[selectedDay],
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
              
              // Update calendar markers
              updateCalendarMarkers();
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
    // Convert ReminderType to ReminderData for the form
    const reminderData: ReminderData = {
      _id: reminder.id,
      title: reminder.title,
      description: reminder.description,
      start_time: parseISO(reminder.date + 'T' + reminder.time).toISOString(),
      // Keep other fields as needed
    };
    
    setEditingReminder(reminderData);
    setFormVisible(true);
  };
  
  // Process voice reminder input
  const processVoiceReminder = async () => {
    if (!voicePrompt.trim()) {
      setAiResponse('Please enter a reminder text');
      return;
    }
  
    setProcessing(true);
    setAiResponse('Processing your request...');
  
    try {
      const result = await reminderService.setVoiceReminder(voicePrompt);
  
      if (result.success) {
        setAiResponse('Reminder set! ' + result.message);
        
        // Refresh the reminders list
        fetchReminders();
        
        // Reset input after successful processing
        setTimeout(() => {
          setVoicePrompt('');
          setProcessing(false);
          setVoiceModalVisible(false);
        }, 3000);
      } else {
        setAiResponse(result.message || 'Failed to process reminder');
        setProcessing(false);
      }
    } catch (error) {
      console.error('Error processing voice reminder:', error);
      setAiResponse('Error connecting to the server');
      setProcessing(false);
    }
  };
  
  // Initialize
  useEffect(() => {
    // Load reminders for the initial date
    fetchReminders();
  }, [fetchReminders]);
  
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
        
        <View style={styles.reminderMetaContainer}>
          {item.recurring && (
            <View style={styles.recurrenceContainer}>
              <Ionicons name="repeat" size={16} color="#757575" />
              <Text style={styles.recurrenceText}>{item.recurrencePattern}</Text>
            </View>
          )}
        </View>
        
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
              Type your reminder in natural language
            </Text>
            <Text style={styles.exampleText}>
              "Remind me to take my medicine at 3 pm"
            </Text>
            
            <TextInput
              style={styles.voiceInput}
              placeholder="Enter your reminder here..."
              value={voicePrompt}
              onChangeText={setVoicePrompt}
              multiline
            />
            
            <Text style={styles.aiResponseText}>{aiResponse}</Text>
            
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
                loading={processing}
                disabled={processing || !voicePrompt.trim()}>
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
          <Menu.Item 
            onPress={() => { 
              setMenuVisible(false);
              setFormVisible(true);
            }} 
            title="Add new reminder" 
            leadingIcon="plus"
          />
          <Menu.Item 
            onPress={() => {
              setMenuVisible(false);
              setVoiceModalVisible(true);
            }} 
            title="Add voice reminder" 
            leadingIcon="microphone"
          />
          <Divider />
        </Menu>
      </View>

      <ScrollView style={styles.scrollView}>
        {/* Calendar section */}
        <Card style={styles.calendarCard}>
          <Card.Content>
            <View style={styles.calendarHeader}>
              <Text style={styles.calendarTitle}>Calendar</Text>
            </View>
            <Calendar
              current={selectedDate}
              onDayPress={handleDateSelect}
              markedDates={markedDates}
              theme={{
                todayTextColor: '#4285F4',
                textDayFontFamily: 'System',
                textMonthFontFamily: 'System',
                textDayHeaderFontFamily: 'System',
                arrowColor: '#4285F4'
              }}
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
  reminderMetaContainer: {
    marginBottom: 8,
    paddingLeft: 52,
  },
  recurrenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
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
    marginTop: 16,
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
  aiResponseText: {
    padding: 12,
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
}); 