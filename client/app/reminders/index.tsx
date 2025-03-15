import React, { useState } from 'react';
import { StyleSheet, View, ScrollView, Dimensions } from 'react-native';
import { Text, Card, Button, IconButton, useTheme, SegmentedButtons, Surface } from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

const { width } = Dimensions.get('window');

type AppRoute = '/dashboard' | '/reminders' | '/talk-to-ai' | '/profile';

export default function RemindersScreen() {
  const theme = useTheme();
  const [timeRange, setTimeRange] = useState('week');

  const handleNavigation = (route: AppRoute) => {
    router.push(route as any); // TODO: Update route types when proper type definitions are available
  };

  // Mock data for medication adherence
  const adherenceData = [85, 90, 100, 95, 88, 92, 96];
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  const maxValue = Math.max(...adherenceData);
  const minValue = Math.min(...adherenceData);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="auto" />
      
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <IconButton icon="arrow-left" onPress={() => router.back()} />
          <Text variant="titleLarge">Health Overview</Text>
        </View>
        <IconButton icon="information" />
      </View>

      <ScrollView style={styles.content}>
        {/* Medication Adherence Tracker */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text variant="titleMedium">Medication Adherence</Text>
            <SegmentedButtons
              value={timeRange}
              onValueChange={setTimeRange}
              buttons={[
                { value: 'week', label: 'Week' },
                { value: 'month', label: 'Month' },
                { value: 'year', label: 'Year' },
              ]}
              style={styles.timeRangeSelector}
            />
          </View>

          <Card style={styles.chartCard}>
            <Card.Content>
              <View style={styles.chartContainer}>
                {adherenceData.map((value, index) => (
                  <View key={index} style={styles.barContainer}>
                    <View style={styles.barWrapper}>
                      <Surface
                        style={[
                          styles.bar,
                          {
                            height: `${value}%`,
                            backgroundColor: value === maxValue ? '#2D68FF' : '#90CAF9',
                          },
                        ]}
                      >
                        <View style={{ flex: 1 }} />
                      </Surface>
                    </View>
                    <Text variant="bodySmall" style={styles.barLabel}>
                      {days[index]}
                    </Text>
                  </View>
                ))}
              </View>
              <View style={styles.statsRow}>
                <View style={styles.stat}>
                  <Text variant="titleLarge" style={styles.statValue}>94%</Text>
                  <Text variant="bodyMedium" style={styles.statLabel}>Average</Text>
                </View>
                <View style={styles.stat}>
                  <Text variant="titleLarge" style={styles.statValue}>100%</Text>
                  <Text variant="bodyMedium" style={styles.statLabel}>Best Day</Text>
                </View>
                <View style={styles.stat}>
                  <Text variant="titleLarge" style={styles.statValue}>85%</Text>
                  <Text variant="bodyMedium" style={styles.statLabel}>Lowest</Text>
                </View>
              </View>
            </Card.Content>
          </Card>
        </View>

        {/* Today's Medications */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text variant="titleMedium">Today's Medications</Text>
            <Button mode="text">View All</Button>
          </View>

          <Card style={styles.medicationCard}>
            <Card.Content>
              <View style={styles.medicationRow}>
                <View style={styles.medicationInfo}>
                  <View style={[styles.pillIcon, { backgroundColor: '#E8F5F1' }]}>
                    <IconButton icon="pill" iconColor="#00B383" />
                  </View>
                  <View>
                    <Text variant="titleMedium">Donepezil</Text>
                    <Text variant="bodyMedium" style={styles.timing}>9:00 AM - 1 pill</Text>
                  </View>
                </View>
                <Button mode="contained" style={styles.takeButton}>
                  Take Now
                </Button>
              </View>
            </Card.Content>
          </Card>

          <Card style={styles.medicationCard}>
            <Card.Content>
              <View style={styles.medicationRow}>
                <View style={styles.medicationInfo}>
                  <View style={[styles.pillIcon, { backgroundColor: '#FFE8EC' }]}>
                    <IconButton icon="pill" iconColor="#FF4471" />
                  </View>
                  <View>
                    <Text variant="titleMedium">Memantine</Text>
                    <Text variant="bodyMedium" style={styles.timing}>2:00 PM - 1 pill</Text>
                  </View>
                </View>
                <Button mode="contained" style={styles.takeButton}>
                  Take Now
                </Button>
              </View>
            </Card.Content>
          </Card>
        </View>

        {/* Bottom Navigation */}
        <Card style={styles.bottomNav}>
          <Card.Content style={styles.bottomNavContent}>
            <IconButton 
              icon="home" 
              size={24}
              onPress={() => handleNavigation('/dashboard')}
            />
            <IconButton
              icon="calendar"
              size={24}
              mode="contained"
              containerColor="#E8F5F1"
              iconColor="#00B383"
              onPress={() => handleNavigation('/reminders')}
            />
            <IconButton 
              icon="robot" 
              size={24}
              onPress={() => handleNavigation('/talk-to-ai')}
            />
            <IconButton 
              icon="cog" 
              size={24}
              onPress={() => handleNavigation('/profile')}
            />
          </Card.Content>
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F6FA',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 4,
    paddingVertical: 8,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  content: {
    flex: 1,
  },
  section: {
    marginBottom: 24,
    paddingHorizontal: 16,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  timeRangeSelector: {
    maxWidth: 200,
    backgroundColor: '#fff',
  },
  chartCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    elevation: 2,
  },
  chartContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    height: 200,
    paddingTop: 20,
    paddingBottom: 20,
  },
  barContainer: {
    flex: 1,
    alignItems: 'center',
  },
  barWrapper: {
    height: '100%',
    width: 20,
    backgroundColor: '#F5F6FA',
    borderRadius: 10,
    overflow: 'hidden',
    justifyContent: 'flex-end',
  },
  bar: {
    width: '100%',
    borderRadius: 10,
  },
  barLabel: {
    marginTop: 8,
    color: '#666',
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  stat: {
    alignItems: 'center',
  },
  statValue: {
    color: '#2D68FF',
    fontWeight: 'bold',
  },
  statLabel: {
    color: '#666',
  },
  medicationCard: {
    backgroundColor: 'white',
    marginBottom: 12,
    borderRadius: 12,
  },
  medicationRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  medicationInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  pillIcon: {
    borderRadius: 12,
  },
  timing: {
    color: '#666',
  },
  takeButton: {
    borderRadius: 8,
  },
  bottomNav: {
    borderRadius: 0,
    elevation: 8,
  },
  bottomNavContent: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 0,
  },
}); 