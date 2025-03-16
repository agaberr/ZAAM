import React, { useState } from 'react';
import { StyleSheet, View, ScrollView, TouchableOpacity, FlatList } from 'react-native';
import { Text, Button, Card, FAB, Chip, Avatar, IconButton, Divider, Menu } from 'react-native-paper';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from '@expo/vector-icons';
import { Calendar } from 'react-native-calendars';
import BottomNavigation from './components/BottomNavigation';

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

export default function RemindersScreen() {
  const router = useRouter();
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [markedDates, setMarkedDates] = useState({
    [selectedDate]: { selected: true, selectedColor: '#4285F4' },
    '2023-06-15': { marked: true, dotColor: '#FF9500' },
    '2023-06-18': { marked: true, dotColor: '#FF3B30' },
    '2023-06-20': { marked: true, dotColor: '#34C759' },
    '2023-06-22': { marked: true, dotColor: '#4285F4' },
  });
  const [filterVisible, setFilterVisible] = useState(false);
  const [activeFilter, setActiveFilter] = useState<ReminderType | 'all'>('all');
  const [menuVisible, setMenuVisible] = useState(false);

  // Sample reminders data
  const reminders: Reminder[] = [
    {
      id: '1',
      title: 'Take Aspirin',
      time: '09:00 AM',
      date: selectedDate,
      type: 'medication',
      description: '100mg - 1 tablet with water',
      completed: false,
      recurring: true,
      recurrencePattern: 'Daily'
    },
    {
      id: '2',
      title: 'Doctor Appointment',
      time: '11:30 AM',
      date: selectedDate,
      type: 'appointment',
      description: 'Follow-up with Dr. Smith',
      location: 'Memorial Hospital, Room 302',
      completed: false
    },
    {
      id: '3',
      title: 'Take Donepezil',
      time: '01:00 PM',
      date: selectedDate,
      type: 'medication',
      description: '5mg - 1 tablet after lunch',
      completed: false,
      recurring: true,
      recurrencePattern: 'Daily'
    },
    {
      id: '4',
      title: 'Memory Exercise',
      time: '03:00 PM',
      date: selectedDate,
      type: 'activity',
      description: 'Complete the daily memory puzzle in the app',
      completed: false,
      recurring: true,
      recurrencePattern: 'Weekdays'
    },
    {
      id: '5',
      title: 'Drink Water',
      time: '04:00 PM',
      date: selectedDate,
      type: 'hydration',
      description: '1 glass of water',
      completed: false,
      recurring: true,
      recurrencePattern: 'Every 2 hours'
    }
  ];

  const filteredReminders = activeFilter === 'all' 
    ? reminders 
    : reminders.filter(reminder => reminder.type === activeFilter);

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

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <IconButton
          icon="arrow-left"
          size={24}
          onPress={() => router.back()}
        />
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
              console.log('Sync with Google Calendar');
              setMenuVisible(false);
            }} 
            title="Sync with Google Calendar" 
            leadingIcon="sync"
          />
          <Menu.Item 
            onPress={() => {
              console.log('Import events');
              setMenuVisible(false);
            }} 
            title="Import events" 
            leadingIcon="import"
          />
          <Menu.Item 
            onPress={() => {
              console.log('Export events');
              setMenuVisible(false);
            }} 
            title="Export events" 
            leadingIcon="export"
          />
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
              <Chip 
                icon="sync" 
                mode="outlined" 
                style={styles.syncChip}
                onPress={() => console.log('Syncing with Google Calendar')}>
                Synced
              </Chip>
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
            Upcoming Reminders for {new Date(selectedDate).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
          </Text>
          
          {filteredReminders.length > 0 ? (
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
                  onPress={() => console.log('Add reminder')}
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
                  onPress={() => router.push('/talk-to-ai')}
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

        {/* Reminder statistics */}
        <Card style={styles.statsCard}>
          <Card.Content>
            <Text style={styles.statsTitle}>Reminder Statistics</Text>
            <View style={styles.statsContainer}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>85%</Text>
                <Text style={styles.statLabel}>Completion Rate</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statValue}>12</Text>
                <Text style={styles.statLabel}>Today's Reminders</Text>
              </View>
              <View style={styles.statDivider} />
              <View style={styles.statItem}>
                <Text style={styles.statValue}>5</Text>
                <Text style={styles.statLabel}>Upcoming</Text>
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* Bottom spacer for navigation bar */}
        <View style={styles.bottomSpacer} />
      </ScrollView>

      {/* FAB for adding new reminders */}
      <FAB
        icon="plus"
        style={styles.fab}
        onPress={() => console.log('Add new reminder')}
        color="white"
      />
      
      {/* Fixed Bottom Navigation */}
      <BottomNavigation />
    </SafeAreaView>
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
}); 