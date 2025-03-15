import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, RefreshControl } from 'react-native';
import { Card, Title, Paragraph, List, Divider, Avatar, Button } from 'react-native-paper';
import { useAuth } from '../../context/AuthContext';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const DashboardScreen = () => {
  const { user } = useAuth();
  const [refreshing, setRefreshing] = useState(false);
  const [activityLogs, setActivityLogs] = useState([]);
  const [upcomingMedications, setUpcomingMedications] = useState([]);
  const [conversationInsights, setConversationInsights] = useState('');

  // Fetch dashboard data
  const fetchDashboardData = async () => {
    setRefreshing(true);
    try {
      // In a real app, you would fetch data from your API
      // For demo purposes, we'll use mock data
      
      // Mock activity logs
      setActivityLogs([
        { id: 1, type: 'conversation', time: '10:30 AM', description: 'Talked about family photos' },
        { id: 2, type: 'reminder', time: '9:00 AM', description: 'Took morning medication' },
        { id: 3, type: 'exercise', time: 'Yesterday', description: 'Completed memory exercises' },
        { id: 4, type: 'appointment', time: 'Yesterday', description: 'Doctor appointment reminder' },
      ]);
      
      // Mock upcoming medications
      setUpcomingMedications([
        { id: 1, name: 'Aspirin', dosage: '100mg', time: '12:00 PM', status: 'pending' },
        { id: 2, name: 'Vitamin D', dosage: '1000 IU', time: '6:00 PM', status: 'pending' },
        { id: 3, name: 'Donepezil', dosage: '5mg', time: '8:00 PM', status: 'pending' },
      ]);
      
      // Mock conversation insights
      setConversationInsights('Based on recent conversations, you\'ve been talking about family memories and showing positive emotional responses to photos from your vacation last year.');
    } catch (error) {
      console.error('Error fetching dashboard data', error);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const onRefresh = () => {
    fetchDashboardData();
  };

  const getActivityIcon = (type) => {
    switch (type) {
      case 'conversation':
        return 'chat';
      case 'reminder':
        return 'bell';
      case 'exercise':
        return 'brain';
      case 'appointment':
        return 'calendar';
      default:
        return 'information';
    }
  };

  return (
    <ScrollView 
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <Card style={styles.welcomeCard}>
        <Card.Content style={styles.welcomeCardContent}>
          <View>
            <Title style={styles.welcomeTitle}>Hello, {user?.full_name?.split(' ')[0] || 'there'}!</Title>
            <Paragraph>Here's your daily summary</Paragraph>
          </View>
          <Avatar.Text 
            size={50} 
            label={user?.full_name?.split(' ').map(n => n[0]).join('') || 'U'} 
            backgroundColor="#6200ee" 
          />
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>Recent Activity</Title>
          {activityLogs.map((activity) => (
            <View key={activity.id}>
              <List.Item
                title={activity.description}
                description={activity.time}
                left={props => <List.Icon {...props} icon={getActivityIcon(activity.type)} />}
              />
              {activity.id !== activityLogs.length && <Divider />}
            </View>
          ))}
          <Button 
            mode="text" 
            onPress={() => {/* Navigate to full activity log */}}
            style={styles.viewMoreButton}
          >
            View More
          </Button>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>Upcoming Medications</Title>
          {upcomingMedications.map((med) => (
            <View key={med.id}>
              <List.Item
                title={med.name}
                description={`${med.dosage} at ${med.time}`}
                left={props => <List.Icon {...props} icon="pill" />}
                right={() => (
                  <Button 
                    mode={med.status === 'completed' ? 'contained' : 'outlined'}
                    onPress={() => {/* Mark as taken */}}
                    style={med.status === 'completed' ? styles.takenButton : styles.pendingButton}
                    labelStyle={med.status === 'completed' ? styles.takenButtonLabel : {}}
                  >
                    {med.status === 'completed' ? 'Taken' : 'Take'}
                  </Button>
                )}
              />
              {med.id !== upcomingMedications.length && <Divider />}
            </View>
          ))}
          <Button 
            mode="text" 
            onPress={() => {/* Navigate to reminders screen */}}
            style={styles.viewMoreButton}
          >
            View All Medications
          </Button>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>AI Conversation Insights</Title>
          <Paragraph style={styles.insightsParagraph}>
            {conversationInsights}
          </Paragraph>
          <Button 
            mode="contained" 
            icon="robot" 
            onPress={() => {/* Navigate to AI interaction screen */}}
            style={styles.talkButton}
          >
            Talk to AI Assistant
          </Button>
        </Card.Content>
      </Card>

      <View style={styles.spacer} />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  welcomeCard: {
    margin: 16,
    marginBottom: 8,
    backgroundColor: '#6200ee',
  },
  welcomeCardContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  welcomeTitle: {
    color: 'white',
    fontSize: 22,
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
  viewMoreButton: {
    alignSelf: 'flex-end',
    marginTop: 5,
  },
  pendingButton: {
    marginVertical: 4,
  },
  takenButton: {
    marginVertical: 4,
    backgroundColor: '#4CAF50',
  },
  takenButtonLabel: {
    color: 'white',
  },
  insightsParagraph: {
    marginBottom: 15,
    lineHeight: 20,
  },
  talkButton: {
    marginTop: 5,
  },
  spacer: {
    height: 20,
  },
});

export default DashboardScreen;
