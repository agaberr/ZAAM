import React from 'react';
import { StyleSheet, View, ScrollView } from 'react-native';
import { Text, Avatar, Card, Button, IconButton, useTheme, Chip } from 'react-native-paper';
import { router } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';

type AppRoute = '/dashboard' | '/reminders' | '/talk-to-ai' | '/profile';

export default function DashboardScreen() {
  const theme = useTheme();

  const handleNavigation = (route: AppRoute) => {
    router.push(route as any); // TODO: Update route types when proper type definitions are available
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.userInfo}>
          <Avatar.Image
            size={40}
            source={require('../assets/default-avatar.png')}
          />
          <View style={styles.greeting}>
            <Text variant="bodyMedium" style={styles.goodMorning}>Good morning!</Text>
            <Text variant="titleMedium" style={styles.userName}>Noah Turner</Text>
          </View>
        </View>
        <Text style={styles.location}>New York, NY</Text>
      </View>

      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <Card style={styles.searchBar}>
          <Card.Content style={styles.searchContent}>
            <IconButton icon="magnify" size={24} />
            <Text variant="bodyMedium" style={styles.searchPlaceholder}>
              Search a doctor, drugs, etc...
            </Text>
            <IconButton icon="microphone" size={24} />
          </Card.Content>
        </Card>
      </View>

      <ScrollView style={styles.content}>
        {/* Recent Appointments */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text variant="titleMedium">Recent Appointments</Text>
            <Button mode="text">View All</Button>
          </View>
          
          <Card style={styles.appointmentCard}>
            <Card.Content>
              <View style={styles.doctorInfo}>
                <Avatar.Image
                  size={50}
                  source={require('../assets/doctor-avatar.png')}
                />
                <View style={styles.appointmentDetails}>
                  <Text variant="titleMedium">Dr. Olivia Carter</Text>
                  <Text variant="bodyMedium" style={styles.specialty}>Neurology</Text>
                  <View style={styles.dateTime}>
                    <Text variant="bodyMedium">24 Apr, Monday</Text>
                    <Text variant="bodyMedium">7:00 am - 7:30 am</Text>
                  </View>
                </View>
              </View>
              <View style={styles.appointmentActions}>
                <Button mode="outlined" style={styles.actionButton}>
                  Re-schedule
                </Button>
                <Button mode="contained" style={styles.actionButton}>
                  View profile
                </Button>
              </View>
            </Card.Content>
          </Card>
        </View>

        {/* Doctors Section */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text variant="titleMedium">Doctors</Text>
            <Button mode="text">View All</Button>
          </View>

          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.categoryScroll}>
            <Chip selected style={styles.categoryChip}>All</Chip>
            <Chip style={styles.categoryChip}>Cardiology</Chip>
            <Chip style={styles.categoryChip}>Dermatology</Chip>
            <Chip style={styles.categoryChip}>Neurology</Chip>
          </ScrollView>

          <Card style={styles.doctorCard}>
            <Card.Content>
              <View style={styles.doctorInfo}>
                <Avatar.Image
                  size={50}
                  source={require('../assets/doctor-avatar.png')}
                />
                <View style={styles.doctorDetails}>
                  <Text variant="titleMedium">Dr. Daniel Reynolds</Text>
                  <Text variant="bodyMedium" style={styles.specialty}>Cardiologist</Text>
                  <View style={styles.rating}>
                    <IconButton icon="star" size={16} iconColor="#FFD700" />
                    <Text>4.9</Text>
                    <Text variant="bodySmall" style={styles.reviews}>(86 Reviews)</Text>
                  </View>
                </View>
                <IconButton
                  icon="message"
                  mode="contained"
                  containerColor={theme.colors.primary}
                  iconColor="white"
                  size={24}
                  style={styles.messageButton}
                />
              </View>
            </Card.Content>
          </Card>
        </View>
      </ScrollView>

      {/* Bottom Navigation */}
      <Card style={styles.bottomNav}>
        <Card.Content style={styles.bottomNavContent}>
          <IconButton
            icon="home"
            size={24}
            mode="contained"
            containerColor="#E8F5F1"
            iconColor="#00B383"
            onPress={() => handleNavigation('/dashboard')}
          />
          <IconButton 
            icon="calendar" 
            size={24}
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
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  userInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  greeting: {
    gap: 4,
  },
  goodMorning: {
    color: '#666',
  },
  userName: {
    fontWeight: 'bold',
  },
  location: {
    color: '#666',
  },
  searchContainer: {
    paddingHorizontal: 16,
    marginBottom: 16,
  },
  searchBar: {
    backgroundColor: 'white',
    elevation: 2,
  },
  searchContent: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 0,
  },
  searchPlaceholder: {
    flex: 1,
    color: '#666',
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
    marginBottom: 12,
  },
  appointmentCard: {
    backgroundColor: '#2D68FF',
    marginBottom: 16,
  },
  doctorInfo: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 16,
  },
  appointmentDetails: {
    flex: 1,
    gap: 4,
  },
  specialty: {
    color: '#666',
    marginBottom: 8,
  },
  dateTime: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  appointmentActions: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    flex: 1,
  },
  categoryScroll: {
    marginBottom: 16,
  },
  categoryChip: {
    marginRight: 8,
  },
  doctorCard: {
    backgroundColor: 'white',
    marginBottom: 12,
  },
  doctorDetails: {
    flex: 1,
    gap: 4,
  },
  rating: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  reviews: {
    color: '#666',
  },
  messageButton: {
    margin: 0,
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