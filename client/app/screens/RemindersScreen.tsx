import React, { useState, useEffect, useCallback } from 'react';
import { StyleSheet, View, ScrollView, TouchableOpacity, FlatList, Alert, Modal, TextInput, Dimensions } from 'react-native';
import { Text, Button, Card, FAB, Chip, Avatar, IconButton, Divider, Menu, Portal, ActivityIndicator } from 'react-native-paper';
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from '@expo/vector-icons';
import { Calendar } from 'react-native-calendars';
import { useFocusEffect } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import ReminderForm from '../components/ReminderForm';
import { reminderService, ReminderData, ReminderType, ReminderStats } from '../services/reminderService';
import { format, parseISO } from 'date-fns';

const { width } = Dimensions.get('window');

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
    [format(new Date(), 'yyyy-MM-dd')]: { selected: true, selectedColor: '#6366F1' },
  });
  
  // UI state
  const [loading, setLoading] = useState(true);
  const [loadingStats, setLoadingStats] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filterVisible, setFilterVisible] = useState(false);
  const [activeFilter, setActiveFilter] = useState<ReminderCategory | 'all'>('all');
  const [menuVisible, setMenuVisible] = useState(false);
  const [completedExpanded, setCompletedExpanded] = useState(false);
  
  // Form state
  const [formVisible, setFormVisible] = useState(false);
  const [editingReminder, setEditingReminder] = useState<ReminderData | undefined>(undefined);
  
  // Data state
  const [reminders, setReminders] = useState<ReminderType[]>([]);
  const [stats, setStats] = useState<ReminderStats>({
    totalCount: 0,
    completedCount: 0,
    pendingCount: 0,
    upcomingCount: 0,
    completionRate: 0,
    byType: {
      medication: 0,
      appointment: 0,
      activity: 0,
      hydration: 0
    }
  });
  
  // Fetch global statistics
  const fetchStatistics = useCallback(async () => {
    try {
      setLoadingStats(true);
      const statsData = await reminderService.getStatistics();
      setStats(statsData);
    } catch (error) {
      console.error('Error fetching statistics:', error);
    } finally {
      setLoadingStats(false);
    }
  }, []);
  
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
      
      // Refresh statistics whenever reminders are fetched
      fetchStatistics();
    } catch (error) {
      console.error('Error fetching reminders:', error);
      Alert.alert('Error', 'Failed to load reminders');
    } finally {
      setLoading(false);
    }
  }, [selectedDate, fetchStatistics]);
  
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
      
      // Refresh the reminders to remove completed items from view
      fetchReminders();
      
      // Refresh statistics after toggling completion
      fetchStatistics();
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
              
              // Refresh reminders from server instead of just updating local state
              await fetchReminders();
              
              // Refresh statistics after deletion
              fetchStatistics();
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
    try {
      // Convert ReminderType to ReminderData for the form
      const reminderData: ReminderData = {
        _id: reminder.id,
        title: reminder.title,
        description: reminder.description,
        start_time: reminder.start_time, // Use the preserved ISO string
        recurrence: reminder.recurring ? 'daily' : null, // Default to daily if recurring
      };
      
      setEditingReminder(reminderData);
      setFormVisible(true);
    } catch (error) {
      console.error('Error preparing reminder for edit:', error);
      Alert.alert('Error', 'Failed to load reminder for editing');
    }
  };
  
  // Initialize
  useEffect(() => {
    // Load reminders for the initial date and get statistics
    fetchReminders();
  }, [fetchReminders]);
  
  // Focus effect to refresh data when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      fetchReminders();
      return () => {};
    }, [fetchReminders])
  );
  
  // Filter reminders based on active filter - separate pending and completed
  const pendingReminders = activeFilter === 'all' 
    ? reminders.filter(reminder => !reminder.completed)
    : reminders.filter(reminder => reminder.type === activeFilter && !reminder.completed);
    
  const completedReminders = activeFilter === 'all' 
    ? reminders.filter(reminder => reminder.completed)
    : reminders.filter(reminder => reminder.type === activeFilter && reminder.completed);
  
  // Get reminders by type from local data (excluding completed)
  const getRemindersCountByType = (type: ReminderCategory): number => {
    return reminders.filter(r => r.type === type && !r.completed).length;
  };
  
  // Render reminder item (works for both pending and completed)
  const renderReminderItem = ({ item }: { item: ReminderType }) => (
    <View style={[styles.modernReminderCard, item.completed && styles.completedReminderCard]}>
      <View style={styles.cardHeader}>
        <View style={styles.leftSection}>
          <View style={[styles.modernTypeIcon, { backgroundColor: getTypeColor(item.type) }]}>
            {getTypeIcon(item.type)}
          </View>
          <View style={styles.reminderDetails}>
            <Text style={[styles.modernTitle, item.completed && styles.completedTitle]}>
              {item.title}
            </Text>
            <View style={styles.timeContainer}>
              <Ionicons name="time-outline" size={14} color="#64748B" />
              <Text style={styles.modernTime}>{item.time}</Text>
            </View>
            {item.description && (
              <Text style={styles.modernDescription} numberOfLines={2}>
                {item.description}
              </Text>
            )}
          </View>
        </View>
        
        <TouchableOpacity 
          style={[styles.modernCompletionButton, item.completed && styles.modernCompletedButton]}
          onPress={() => toggleReminderCompletion(item.id, item.completed)}
          activeOpacity={0.8}
        >
          {item.completed ? (
            <Ionicons name="checkmark" size={20} color="white" />
          ) : (
            <View style={styles.incompleteDot} />
          )}
        </TouchableOpacity>
      </View>
      
      {item.recurring && (
        <View style={styles.recurrenceBadge}>
          <Ionicons name="repeat" size={12} color="#6366F1" />
          <Text style={styles.recurrenceBadgeText}>{item.recurrencePattern}</Text>
        </View>
      )}
      
      <View style={styles.modernActions}>
        <TouchableOpacity 
          style={styles.modernActionButton}
          onPress={() => handleEditReminder(item)}
          activeOpacity={0.7}
        >
          <Ionicons name="pencil" size={16} color="#6366F1" />
          <Text style={styles.modernActionText}>Edit</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.modernActionButton, styles.deleteAction]}
          onPress={() => handleDeleteReminder(item.id)}
          activeOpacity={0.7}
        >
          <Ionicons name="trash-outline" size={16} color="#EF4444" />
          <Text style={[styles.modernActionText, styles.deleteActionText]}>Delete</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
  
  return (
    <View style={styles.container}>
      {/* Modern Header with Solid Blue */}
      <View 
        style={styles.headerContainer}
      >
        <View style={styles.headerBackground}>
          <View style={styles.headerContent}>
            <View style={styles.headerLeft}>
              <Text style={styles.headerTitle}>Reminders</Text>
              <Text style={styles.headerSubtitle}>
                {format(new Date(selectedDate), 'EEEE, MMM d')}
              </Text>
            </View>
            
            <View style={styles.headerRight}>
              <TouchableOpacity
                style={styles.addButton}
                onPress={() => {
                  setEditingReminder(undefined);
                  setFormVisible(true);
                }}
                activeOpacity={0.8}
              >
                <Ionicons name="add" size={24} color="white" />
              </TouchableOpacity>
              
              <Menu
                visible={menuVisible}
                onDismiss={() => setMenuVisible(false)}
                anchor={
                  <TouchableOpacity
                    style={styles.menuButton}
                    onPress={() => setMenuVisible(true)}
                    activeOpacity={0.8}
                  >
                    <Ionicons name="ellipsis-vertical" size={20} color="white" />
                  </TouchableOpacity>
                }
                contentStyle={styles.menuContent}
              >
                <Menu.Item 
                  onPress={() => { 
                    setMenuVisible(false);
                    fetchReminders();
                  }} 
                  title="Refresh" 
                  leadingIcon="refresh"
                />
                <Divider />
                <Menu.Item 
                  onPress={() => { 
                    setMenuVisible(false);
                    fetchStatistics();
                  }} 
                  title="Update Stats" 
                  leadingIcon="chart-line"
                />
              </Menu>
            </View>
          </View>
        </View>
      </View>

      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {/* Stats Overview */}
        <View 
          style={styles.statsSection}
        >
          <View style={styles.statsHeader}>
            <Text style={styles.sectionTitle}>Today's Overview</Text>
          </View>
          
          <View style={styles.statsGrid}>
            <View style={[styles.statCard, styles.statCard1]}>
              <View style={styles.statIconContainer}>
                <Ionicons name="checkmark-circle" size={24} color="#10B981" />
              </View>
              <Text style={styles.statValue}>{stats.completedCount}</Text>
              <Text style={styles.statLabel}>Completed</Text>
            </View>
            
            <View style={[styles.statCard, styles.statCard2]}>
              <View style={styles.statIconContainer}>
                <Ionicons name="time" size={24} color="#F59E0B" />
              </View>
              <Text style={styles.statValue}>{stats.pendingCount}</Text>
              <Text style={styles.statLabel}>Pending</Text>
            </View>
            
            <View style={[styles.statCard, styles.statCard3]}>
              <View style={styles.statIconContainer}>
                <Ionicons name="trending-up" size={24} color="#6366F1" />
              </View>
              <Text style={styles.statValue}>{stats.completionRate}%</Text>
              <Text style={styles.statLabel}>Success Rate</Text>
            </View>
          </View>
        </View>

        {/* Filter Chips */}
        <View 
          style={styles.filterSection}
        >
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filtersScrollView}>
            <TouchableOpacity
              style={[
                styles.modernFilterChip,
                activeFilter === 'all' && styles.activeFilterChip
              ]}
              onPress={() => setActiveFilter('all')}
            >
              <Text style={[
                styles.filterChipText,
                activeFilter === 'all' && styles.activeFilterChipText
              ]}>
                All ({reminders.filter(r => !r.completed).length})
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[
                styles.modernFilterChip,
                activeFilter === 'medication' && styles.activeFilterChip
              ]}
              onPress={() => setActiveFilter('medication')}
            >
              <Ionicons name="medical" size={16} color={activeFilter === 'medication' ? 'white' : '#64748B'} />
              <Text style={[
                styles.filterChipText,
                activeFilter === 'medication' && styles.activeFilterChipText
              ]}>
                Medication ({getRemindersCountByType('medication')})
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[
                styles.modernFilterChip,
                activeFilter === 'appointment' && styles.activeFilterChip
              ]}
              onPress={() => setActiveFilter('appointment')}
            >
              <Ionicons name="calendar" size={16} color={activeFilter === 'appointment' ? 'white' : '#64748B'} />
              <Text style={[
                styles.filterChipText,
                activeFilter === 'appointment' && styles.activeFilterChipText
              ]}>
                Appointments ({getRemindersCountByType('appointment')})
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[
                styles.modernFilterChip,
                activeFilter === 'activity' && styles.activeFilterChip
              ]}
              onPress={() => setActiveFilter('activity')}
            >
              <Ionicons name="fitness" size={16} color={activeFilter === 'activity' ? 'white' : '#64748B'} />
              <Text style={[
                styles.filterChipText,
                activeFilter === 'activity' && styles.activeFilterChipText
              ]}>
                Activities ({getRemindersCountByType('activity')})
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[
                styles.modernFilterChip,
                activeFilter === 'hydration' && styles.activeFilterChip
              ]}
              onPress={() => setActiveFilter('hydration')}
            >
              <Ionicons name="water" size={16} color={activeFilter === 'hydration' ? 'white' : '#64748B'} />
              <Text style={[
                styles.filterChipText,
                activeFilter === 'hydration' && styles.activeFilterChipText
              ]}>
                Hydration ({getRemindersCountByType('hydration')})
              </Text>
            </TouchableOpacity>
          </ScrollView>
        </View>

        {/* Calendar Section */}
        <View 
          style={styles.calendarSection}
        >
          <View style={styles.calendarCard}>
            <Calendar
              current={selectedDate}
              onDayPress={handleDateSelect}
              markedDates={markedDates}
              theme={{
                backgroundColor: 'white',
                calendarBackground: 'white',
                textSectionTitleColor: '#6366F1',
                selectedDayBackgroundColor: '#6366F1',
                selectedDayTextColor: 'white',
                todayTextColor: '#6366F1',
                dayTextColor: '#1F2937',
                textDisabledColor: '#D1D5DB',
                dotColor: '#6366F1',
                selectedDotColor: 'white',
                arrowColor: '#6366F1',
                disabledArrowColor: '#D1D5DB',
                monthTextColor: '#1F2937',
                indicatorColor: '#6366F1',
                textDayFontFamily: 'System',
                textMonthFontFamily: 'System',
                textDayHeaderFontFamily: 'System',
                textDayFontWeight: '400',
                textMonthFontWeight: '600',
                textDayHeaderFontWeight: '500',
                textDayFontSize: 16,
                textMonthFontSize: 18,
                textDayHeaderFontSize: 14
              }}
            />
          </View>
        </View>

        {/* Reminders List */}
        <View 
          style={styles.remindersSection}
        >
          <View style={styles.remindersHeader}>
            <Text style={styles.sectionTitle}>
              {format(new Date(selectedDate), 'EEEE, MMMM d')}
            </Text>
            <Text style={styles.reminderCount}>
              {pendingReminders.length + completedReminders.length} reminder{pendingReminders.length + completedReminders.length !== 1 ? 's' : ''}
            </Text>
          </View>
          
          {loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#6366F1" />
              <Text style={styles.loadingText}>Loading reminders...</Text>
            </View>
          ) : pendingReminders.length + completedReminders.length > 0 ? (
            <>
              {/* Pending Reminders Section */}
              {pendingReminders.length > 0 ? (
                <View style={styles.reminderSubsection}>
                  <View style={styles.subsectionHeader}>
                    <Text style={styles.subsectionTitle}>Pending</Text>
                    <Text style={styles.subsectionCount}>
                      {pendingReminders.length} reminder{pendingReminders.length !== 1 ? 's' : ''}
                    </Text>
                  </View>
                  <FlatList
                    data={pendingReminders}
                    renderItem={renderReminderItem}
                    keyExtractor={item => item.id}
                    scrollEnabled={false}
                    contentContainerStyle={styles.remindersList}
                    showsVerticalScrollIndicator={false}
                  />
                </View>
              ) : (
                <View style={styles.addReminderPrompt}>
                  <View style={styles.addReminderContent}>
                    <View style={styles.addReminderIcon}>
                      <Ionicons name="calendar-outline" size={60} color="#CBD5E1" />
                    </View>
                    <Text style={styles.addReminderTitle}>No reminders yet</Text>
                    <Text style={styles.addReminderText}>
                      Start by adding your first reminder for this day
                    </Text>
                    <TouchableOpacity
                      style={styles.addReminderButton}
                      onPress={() => {
                        setEditingReminder(undefined);
                        setFormVisible(true);
                      }}
                      activeOpacity={0.8}
                    >
                      <Ionicons name="add" size={20} color="white" />
                      <Text style={styles.addReminderButtonText}>Add Reminder</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              )}
              
              {/* Completed Reminders Section */}
              {completedReminders.length > 0 && (
                <View style={styles.reminderSubsection}>
                  <TouchableOpacity 
                    style={styles.subsectionHeader}
                    onPress={() => setCompletedExpanded(!completedExpanded)}
                    activeOpacity={0.7}
                  >
                    <View style={styles.subsectionHeaderLeft}>
                      <Text style={styles.subsectionTitle}>Completed</Text>
                      <Text style={styles.subsectionCount}>
                        {completedReminders.length} reminder{completedReminders.length !== 1 ? 's' : ''}
                      </Text>
                    </View>
                    <Ionicons 
                      name={completedExpanded ? "chevron-up" : "chevron-down"} 
                      size={20} 
                      color="#64748B" 
                    />
                  </TouchableOpacity>
                  
                  {completedExpanded && (
                    <FlatList
                      data={completedReminders}
                      renderItem={renderReminderItem}
                      keyExtractor={item => item.id}
                      scrollEnabled={false}
                      contentContainerStyle={styles.remindersList}
                      showsVerticalScrollIndicator={false}
                    />
                  )}
                </View>
              )}
            </>
          ) : (
            <View style={styles.emptyStateCard}>
              <View style={styles.emptyStateContent}>
                <View style={styles.emptyStateIcon}>
                  <Ionicons name="calendar-outline" size={60} color="#CBD5E1" />
                </View>
                <Text style={styles.emptyStateTitle}>No reminders yet</Text>
                <Text style={styles.emptyStateText}>
                  Start by adding your first reminder for this day
                </Text>
                <TouchableOpacity
                  style={styles.emptyStateButton}
                  onPress={() => {
                    setEditingReminder(undefined);
                    setFormVisible(true);
                  }}
                  activeOpacity={0.8}
                >
                  <Ionicons name="add" size={20} color="white" />
                  <Text style={styles.emptyStateButtonText}>Add Reminder</Text>
                </TouchableOpacity>
              </View>
            </View>
          )}
        </View>
      </ScrollView>

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
  headerContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 100,
    height: 140,
  },
  headerBackground: {
    backgroundColor: '#6366F1',
    paddingTop: 50, // Account for status bar
    paddingBottom: 20,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    shadowColor: '#6366F1',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  headerLeft: {
    flexDirection: 'column',
    flex: 1,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: 'white',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  addButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  menuButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  menuContent: {
    marginTop: 60,
    borderRadius: 12,
  },
  scrollView: {
    flex: 1,
    paddingTop: 140, // Space for header
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
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#1F2937',
  },
  remindersSection: {
    marginHorizontal: 16,
    marginBottom: 16,
  },
  remindersList: {
    gap: 12,
  },
  modernReminderCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 4,
    borderWidth: 1,
    borderColor: '#F1F5F9',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  leftSection: {
    flexDirection: 'row',
    flex: 1,
  },
  modernTypeIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  reminderDetails: {
    flex: 1,
  },
  modernTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 6,
    lineHeight: 24,
  },
  timeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  modernTime: {
    fontSize: 14,
    fontWeight: '500',
    color: '#64748B',
    marginLeft: 6,
  },
  modernDescription: {
    fontSize: 14,
    color: '#64748B',
    lineHeight: 20,
    marginTop: 4,
  },
  modernCompletionButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: '#6366F1',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white',
    shadowColor: '#6366F1',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
  },
  incompleteDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#6366F1',
  },
  recurrenceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#EEF2FF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    alignSelf: 'flex-start',
    marginBottom: 12,
  },
  recurrenceBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6366F1',
    marginLeft: 4,
  },
  modernActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 8,
  },
  modernActionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#F8FAFC',
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  modernActionText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6366F1',
    marginLeft: 4,
  },
  deleteAction: {
    backgroundColor: '#FEF2F2',
    borderColor: '#FECACA',
  },
  deleteActionText: {
    color: '#EF4444',
  },
  emptyStateCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  emptyStateContent: {
    alignItems: 'center',
    paddingVertical: 40,
    paddingHorizontal: 20,
  },
  emptyStateIcon: {
    marginBottom: 16,
  },
  emptyStateTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 8,
    textAlign: 'center',
  },
  emptyStateText: {
    fontSize: 14,
    color: '#64748B',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 24,
  },
  emptyStateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#6366F1',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
    shadowColor: '#6366F1',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  emptyStateButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: 'white',
    marginLeft: 8,
  },
  loadingContainer: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  loadingText: {
    fontSize: 16,
    color: '#6366F1',
    marginTop: 12,
    fontWeight: '500',
  },
  statsSection: {
    marginHorizontal: 16,
    marginBottom: 16,
  },
  statsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  statCard: {
    flex: 1,
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statCard1: {
    borderTopWidth: 3,
    borderTopColor: '#10B981',
  },
  statCard2: {
    borderTopWidth: 3,
    borderTopColor: '#F59E0B',
  },
  statCard3: {
    borderTopWidth: 3,
    borderTopColor: '#6366F1',
  },
  statIconContainer: {
    marginBottom: 8,
    padding: 8,
    borderRadius: 8,
    backgroundColor: '#F8FAFC',
  },
  statValue: {
    fontSize: 20,
    fontWeight: '800',
    color: '#1F2937',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#64748B',
    fontWeight: '500',
    textAlign: 'center',
  },
  filterSection: {
    marginHorizontal: 16,
    marginBottom: 16,
  },
  modernFilterChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    backgroundColor: '#F1F5F9',
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  activeFilterChip: {
    backgroundColor: '#6366F1',
    borderColor: '#6366F1',
  },
  filterChipText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#64748B',
    marginLeft: 4,
  },
  activeFilterChipText: {
    color: 'white',
  },
  calendarSection: {
    marginHorizontal: 16,
    marginBottom: 20,
  },
  calendarCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  remindersHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  reminderCount: {
    fontSize: 14,
    color: '#64748B',
    fontWeight: '500',
  },
  completedReminderCard: {
    backgroundColor: '#F8FAFC',
  },
  completedTitle: {
    color: '#6366F1',
  },
  modernCompletedButton: {
    backgroundColor: '#6366F1',
  },
  reminderSubsection: {
    marginBottom: 16,
  },
  subsectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  subsectionHeaderLeft: {
    flexDirection: 'column',
    flex: 1,
  },
  subsectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 2,
  },
  subsectionCount: {
    fontSize: 14,
    color: '#64748B',
    fontWeight: '500',
  },
  addReminderPrompt: {
    backgroundColor: 'white',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  addReminderContent: {
    alignItems: 'center',
    paddingVertical: 40,
    paddingHorizontal: 20,
  },
  addReminderIcon: {
    marginBottom: 16,
  },
  addReminderTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 8,
    textAlign: 'center',
  },
  addReminderText: {
    fontSize: 14,
    color: '#64748B',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 24,
  },
  addReminderButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#6366F1',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
    shadowColor: '#6366F1',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  addReminderButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: 'white',
    marginLeft: 8,
  },
}); 