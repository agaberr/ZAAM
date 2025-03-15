import React from 'react';
import { StyleSheet, View, ScrollView, Dimensions } from 'react-native';
import { Text, Card, IconButton, Surface, Button, Avatar, List } from 'react-native-paper';
import { StatusBar } from 'expo-status-bar';
import { router } from 'expo-router';

const { width } = Dimensions.get('window');

export default function DashboardScreen() {
  // Mock data - will be replaced with real data from backend
  const recentActivities = [
    { id: 1, type: 'conversation', time: '10:30 AM', description: 'Morning chat with AI' },
    { id: 2, type: 'medication', time: '9:00 AM', description: 'Took morning medication' },
    { id: 3, type: 'appointment', time: 'Tomorrow', description: 'Doctor appointment' },
  ];

  const upcomingMedications = [
    { id: 1, name: 'Aspirin', time: '2:00 PM', dosage: '100mg' },
    { id: 2, name: 'Vitamin D', time: '8:00 PM', dosage: '1000 IU' },
  ];

  return (
    <View style={styles.container}>
      <StatusBar style="light" />
      
      {/* Header */}
      <Surface style={styles.header} elevation={2}>
        <View style={styles.headerContent}>
          <View>
            <Text variant="titleLarge" style={styles.headerTitle}>Good Morning</Text>
            <Text variant="bodyLarge" style={styles.headerSubtitle}>John Doe</Text>
          </View>
          <IconButton
            icon="account"
            size={24}
            onPress={() => router.push('/profile')}
            mode="contained"
          />
        </View>
      </Surface>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Quick Actions */}
        <View style={styles.quickActions}>
          <Button
            mode="contained"
            icon="robot"
            onPress={() => router.push('/talk-to-ai')}
            style={[styles.quickActionButton, { backgroundColor: '#007AFF' }]}
            contentStyle={styles.quickActionContent}
          >
            Talk to AI
          </Button>
          <Button
            mode="contained"
            icon="calendar"
            onPress={() => router.push('/reminders')}
            style={[styles.quickActionButton, { backgroundColor: '#5856D6' }]}
            contentStyle={styles.quickActionContent}
          >
            Reminders
          </Button>
        </View>

        {/* Activity Summary */}
        <Card style={styles.card}>
          <Card.Title title="Today's Activities" />
          <Card.Content>
            {recentActivities.map((activity) => (
              <List.Item
                key={activity.id}
                title={activity.description}
                description={activity.time}
                left={props => (
                  <List.Icon
                    {...props}
                    icon={
                      activity.type === 'conversation' ? 'chat' :
                      activity.type === 'medication' ? 'pill' : 'calendar'
                    }
                  />
                )}
              />
            ))}
          </Card.Content>
        </Card>

        {/* Medications Tracking */}
        <Card style={styles.card}>
          <Card.Title title="Upcoming Medications" />
          <Card.Content>
            {upcomingMedications.map((med) => (
              <List.Item
                key={med.id}
                title={med.name}
                description={`${med.time} - ${med.dosage}`}
                left={props => <List.Icon {...props} icon="pill" />}
                right={props => (
                  <Button
                    {...props}
                    mode="text"
                    onPress={() => console.log('Marked as taken')}
                  >
                    Take
                  </Button>
                )}
              />
            ))}
          </Card.Content>
        </Card>

        {/* AI Interaction Summary */}
        <Card style={styles.card}>
          <Card.Title title="Recent AI Conversations" />
          <Card.Content>
            <View style={styles.aiSummary}>
              <Avatar.Icon size={40} icon="robot" style={styles.aiIcon} />
              <View style={styles.aiStats}>
                <Text variant="bodyLarge">Last conversation: 2 hours ago</Text>
                <Text variant="bodyMedium">Mood detected: Positive</Text>
              </View>
            </View>
            <Button
              mode="outlined"
              onPress={() => router.push('/talk-to-ai')}
              style={styles.startChatButton}
            >
              Start New Conversation
            </Button>
          </Card.Content>
        </Card>

        {/* Bottom Spacing */}
        <View style={styles.bottomSpacing} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#007AFF',
    paddingTop: 60,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    color: '#fff',
    fontWeight: 'bold',
  },
  headerSubtitle: {
    color: '#fff',
    opacity: 0.8,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  quickActions: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  quickActionButton: {
    flex: 1,
    borderRadius: 12,
  },
  quickActionContent: {
    height: 48,
  },
  card: {
    marginBottom: 16,
    borderRadius: 12,
    elevation: 2,
  },
  aiSummary: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  aiIcon: {
    backgroundColor: '#007AFF',
    marginRight: 16,
  },
  aiStats: {
    flex: 1,
  },
  startChatButton: {
    marginTop: 8,
  },
  bottomSpacing: {
    height: 20,
  },
}); 