import React, { useState } from 'react';
import { StyleSheet, View, ScrollView, Image, Dimensions, TouchableOpacity } from 'react-native';
import { Text, Button, Searchbar, Card, Avatar, IconButton } from 'react-native-paper';
import { Link, useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from '@expo/vector-icons';
import BottomNavigation from '../components/BottomNavigation';

const { width } = Dimensions.get('window');

export default function HomeScreen() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = React.useState('');
  const [showDetails, setShowDetails] = useState(false);

  const onChangeSearch = (query: string) => setSearchQuery(query);

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView style={styles.scrollView}>
          {/* Header with greeting and profile */}
          <View style={styles.header}>
            <View>
              <Text style={styles.greeting}>Hello, Amit</Text>
            </View>
            <TouchableOpacity onPress={() => router.push('/profile')}>
              <Avatar.Icon 
                size={40} 
                icon="account" 
                style={styles.profileIcon}
                color="#000"
              />
            </TouchableOpacity>
          </View>

          {/* Search bar */}
          <View style={styles.searchContainer}>
            <Searchbar
              placeholder="Search..."
              onChangeText={onChangeSearch}
              value={searchQuery}
              style={styles.searchBar}
              iconColor="#777"
            />
          </View>

          {/* Membership card */}
          <Card style={styles.membershipCard}>
            <Card.Content style={styles.membershipContent}>
              <View style={styles.membershipTextContainer}>
                <Text style={styles.membershipTitle}>Buy your Plus membership</Text>
                <View style={styles.priceContainer}>
                  <Text style={styles.originalPrice}>₹1399</Text>
                  <Text style={styles.discountedPrice}>₹699</Text>
                </View>
                <Button 
                  mode="contained" 
                  style={styles.buyButton}
                  labelStyle={styles.buyButtonLabel}
                  onPress={() => {}}>
                  Buy now
                </Button>
              </View>
              <View style={styles.membershipImageContainer}>
                <View style={styles.discountBadge}>
                  <Text style={styles.discountText}>50% off</Text>
                </View>
                <MaterialCommunityIcons name="shield-plus" size={50} color="#4285F4" style={styles.shieldIcon} />
              </View>
            </Card.Content>
          </Card>

          {/* Quick action buttons */}
          <View style={styles.quickActionsContainer}>
            <TouchableOpacity style={styles.quickActionItem} onPress={() => router.push('/talk-to-ai')}>
              <View style={styles.quickActionIconContainer}>
                <FontAwesome5 name="stethoscope" size={24} color="#4285F4" />
              </View>
              <Text style={styles.quickActionText}>Consult AI</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.quickActionItem} onPress={() => router.push('/dashboard/activity')}>
              <View style={styles.quickActionIconContainer}>
                <MaterialCommunityIcons name="shield-check" size={24} color="#4285F4" />
              </View>
              <Text style={styles.quickActionText}>Activity Logs</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.quickActionItem} onPress={() => router.push('/reminders')}>
              <View style={styles.quickActionIconContainer}>
                <MaterialCommunityIcons name="chart-line" size={24} color="#4285F4" />
              </View>
              <Text style={styles.quickActionText}>Reminders</Text>
            </TouchableOpacity>
          </View>

          {/* Health monitoring banner */}
          <TouchableOpacity 
            style={styles.monitoringBanner}
            onPress={() => router.push('/dashboard/health-tracking')}>
            <Text style={styles.monitoringText}>Start monitoring your health today</Text>
            <Ionicons name="chevron-forward" size={20} color="#4285F4" />
          </TouchableOpacity>

          {/* Top articles section */}
          <View style={styles.articlesSection}>
            <Text style={styles.sectionTitle}>Daily Routine Summary</Text>
            
            <View style={styles.articlesContainer}>
              <TouchableOpacity 
                style={styles.articleCard}
                onPress={() => router.push('/dashboard/medications')}>
                <View style={styles.articleContent}>
                  <Text style={styles.articleTitle}>Medications</Text>
                  <Text style={styles.articleDescription}>
                    You have 2 medications scheduled for today
                  </Text>
                </View>
                <View style={styles.articleImageContainer}>
                  <MaterialCommunityIcons name="pill" size={40} color="#4285F4" />
                </View>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={styles.articleCard}
                onPress={() => router.push('/insights')}>
                <View style={styles.articleContent}>
                  <Text style={styles.articleTitle}>Insights</Text>
                  <Text style={styles.articleDescription}>
                    View conversation analysis from your AI companion
                  </Text>
                </View>
                <View style={styles.articleImageContainer}>
                  <MaterialCommunityIcons name="brain" size={40} color="#4285F4" />
                </View>
              </TouchableOpacity>
            </View>
          </View>

          {/* Health tracking section */}
          <View style={styles.articlesSection}>
            <Text style={styles.sectionTitle}>Health & Medication Tracking</Text>
            
            <Card style={styles.medicationCard}>
              <Card.Content>
                <View style={styles.medicationHeader}>
                  <Text style={styles.medicationTitle}>Upcoming Medications</Text>
                  <Button 
                    mode="text" 
                    onPress={() => router.push('/reminders')}
                    labelStyle={{color: '#4285F4'}}>
                    View All
                  </Button>
                </View>
                
                <View style={styles.medicationItem}>
                  <View style={styles.medicationTimeContainer}>
                    <Text style={styles.medicationTime}>9:00 AM</Text>
                  </View>
                  <View style={styles.medicationDetails}>
                    <Text style={styles.medicationName}>Aspirin</Text>
                    <Text style={styles.medicationDosage}>100mg - 1 tablet</Text>
                  </View>
                  <Button 
                    mode="contained" 
                    compact 
                    style={styles.takeMedButton}
                    labelStyle={styles.takeMedButtonLabel}
                    onPress={() => {}}>
                    Take
                  </Button>
                </View>
                
                <View style={styles.medicationItem}>
                  <View style={styles.medicationTimeContainer}>
                    <Text style={styles.medicationTime}>1:00 PM</Text>
                  </View>
                  <View style={styles.medicationDetails}>
                    <Text style={styles.medicationName}>Donepezil</Text>
                    <Text style={styles.medicationDosage}>5mg - 1 tablet</Text>
                  </View>
                  <Button 
                    mode="contained" 
                    compact 
                    style={styles.takeMedButton}
                    labelStyle={styles.takeMedButtonLabel}
                    onPress={() => {}}>
                    Take
                  </Button>
                </View>
              </Card.Content>
            </Card>
          </View>

          {/* Activity logs section */}
          <View style={styles.articlesSection}>
            <Text style={styles.sectionTitle}>Activity Logs</Text>
            
            <Card style={styles.activityCard}>
              <Card.Content>
                <View style={styles.activityItem}>
                  <View style={styles.activityIconContainer}>
                    <MaterialCommunityIcons name="message-text" size={24} color="#4285F4" />
                  </View>
                  <View style={styles.activityDetails}>
                    <Text style={styles.activityTitle}>AI Conversation</Text>
                    <Text style={styles.activityTime}>Today, 10:30 AM</Text>
                    <Text style={styles.activityDescription}>
                      You asked about today's weather and your medication schedule
                    </Text>
                  </View>
                </View>
                
                <View style={styles.activityItem}>
                  <View style={styles.activityIconContainer}>
                    <MaterialCommunityIcons name="pill" size={24} color="#4285F4" />
                  </View>
                  <View style={styles.activityDetails}>
                    <Text style={styles.activityTitle}>Medication Taken</Text>
                    <Text style={styles.activityTime}>Today, 9:05 AM</Text>
                    <Text style={styles.activityDescription}>
                      You took Aspirin (100mg)
                    </Text>
                  </View>
                </View>
                
                <Button 
                  mode="text" 
                  onPress={() => router.push('/dashboard/activity')}
                  style={styles.viewAllButton}
                  labelStyle={{color: '#4285F4'}}>
                  View All Activities
                </Button>
              </Card.Content>
            </Card>
          </View>

          {/* Bottom spacer for navigation bar */}
          <View style={styles.bottomSpacer} />
        </ScrollView>

        {/* Fixed Bottom Navigation */}
        <BottomNavigation />
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  safeArea: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 15,
  },
  greeting: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  profileIcon: {
    backgroundColor: '#E8E8E8',
  },
  searchContainer: {
    paddingHorizontal: 20,
    marginBottom: 15,
  },
  searchBar: {
    borderRadius: 30,
    backgroundColor: 'white',
    elevation: 0,
  },
  membershipCard: {
    marginHorizontal: 20,
    marginBottom: 15,
    borderRadius: 15,
    backgroundColor: '#E8F1FF',
    elevation: 2,
  },
  membershipContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  membershipTextContainer: {
    flex: 2,
  },
  membershipTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  priceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  originalPrice: {
    fontSize: 16,
    textDecorationLine: 'line-through',
    color: '#777',
    marginRight: 8,
  },
  discountedPrice: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  buyButton: {
    backgroundColor: '#0A2540',
    borderRadius: 20,
    width: 100,
    height: 36,
    justifyContent: 'center',
  },
  buyButtonLabel: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  membershipImageContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  discountBadge: {
    position: 'absolute',
    top: -10,
    right: -10,
    backgroundColor: '#777',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 10,
  },
  discountText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  shieldIcon: {
    marginTop: 5,
  },
  quickActionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    marginBottom: 15,
  },
  quickActionItem: {
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
    padding: 15,
    borderRadius: 15,
    width: width / 3.5,
  },
  quickActionIconContainer: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#E8F1FF',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  quickActionText: {
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
  monitoringBanner: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#E8F1FF',
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 15,
    borderRadius: 15,
  },
  monitoringText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#0A2540',
  },
  articlesSection: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginHorizontal: 20,
    marginBottom: 15,
  },
  articlesContainer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    justifyContent: 'space-between',
  },
  articleCard: {
    backgroundColor: '#F5F5F5',
    borderRadius: 15,
    padding: 15,
    width: width / 2.3,
    flexDirection: 'row',
  },
  articleContent: {
    flex: 2,
  },
  articleTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  articleDescription: {
    fontSize: 12,
    color: '#777',
    lineHeight: 16,
  },
  articleImageContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  medicationCard: {
    marginHorizontal: 20,
    borderRadius: 15,
  },
  medicationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  medicationTitle: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  medicationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
    backgroundColor: '#F9F9F9',
    padding: 10,
    borderRadius: 10,
  },
  medicationTimeContainer: {
    backgroundColor: '#E8F1FF',
    padding: 8,
    borderRadius: 8,
    marginRight: 10,
  },
  medicationTime: {
    fontSize: 14,
    fontWeight: '500',
    color: '#4285F4',
  },
  medicationDetails: {
    flex: 1,
  },
  medicationName: {
    fontSize: 16,
    fontWeight: '500',
  },
  medicationDosage: {
    fontSize: 14,
    color: '#777',
  },
  takeMedButton: {
    backgroundColor: '#4285F4',
    borderRadius: 20,
  },
  takeMedButtonLabel: {
    fontSize: 12,
  },
  activityCard: {
    marginHorizontal: 20,
    borderRadius: 15,
  },
  activityItem: {
    flexDirection: 'row',
    marginBottom: 15,
  },
  activityIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#E8F1FF',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  activityDetails: {
    flex: 1,
  },
  activityTitle: {
    fontSize: 16,
    fontWeight: '500',
  },
  activityTime: {
    fontSize: 12,
    color: '#777',
    marginBottom: 5,
  },
  activityDescription: {
    fontSize: 14,
    color: '#333',
  },
  viewAllButton: {
    alignSelf: 'center',
    marginTop: 5,
  },
  bottomSpacer: {
    height: 80,
  },
  headerButton: {
    marginLeft: 10,
  },
}); 