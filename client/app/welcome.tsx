import React from 'react';
import { StyleSheet, View, Dimensions, Pressable, Alert } from 'react-native';
import { Button, Text } from 'react-native-paper';
import { Link, Stack } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from './context/AuthContext';

const { width, height } = Dimensions.get('window');

export default function WelcomeScreen() {

  return (
    <View style={styles.container}>
      {/* Main gradient background */}
      <LinearGradient
        colors={['#1e3c72', '#2a5298', '#4facfe']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.backgroundGradient}
      />
      
      {/* Decorative geometric shapes */}
      <View style={styles.decorativeShapes}>
        <View style={[styles.circle, styles.circle1]} />
        <View style={[styles.circle, styles.circle2]} />
        <View style={[styles.circle, styles.circle3]} />
        <LinearGradient
          colors={['rgba(255, 255, 255, 0.1)', 'rgba(255, 255, 255, 0.05)']}
          style={[styles.shape, styles.shape1]}
        />
        <LinearGradient
          colors={['rgba(255, 255, 255, 0.08)', 'rgba(255, 255, 255, 0.03)']}
          style={[styles.shape, styles.shape2]}
        />
      </View>

      {/* Content overlay */}
      <LinearGradient
        colors={['rgba(0, 0, 0, 0.1)', 'rgba(0, 0, 0, 0.05)', 'rgba(0, 0, 0, 0.1)']}
        style={styles.contentOverlay}
      />

      <View style={styles.content}>
        <View style={styles.titleSection}>
          <Text variant="displaySmall" style={styles.title}>
            Welcome to
          </Text>
          <Text variant="displayLarge" style={styles.brandName}>
            ZAAM
          </Text>
          <Text variant="titleMedium" style={styles.subtitle}>
            Your AI Companion for Alzheimer's Care
          </Text>
        </View>
        
        <View style={styles.buttonContainer}>
          <Link href="/auth/signup" asChild>
            <Button
              mode="contained"
              style={styles.button}
              contentStyle={styles.buttonContent}
              labelStyle={styles.buttonLabel}
            >
              Get Started
            </Button>
          </Link>
          <Link href="/auth/signin" asChild>
            <Button
              mode="text"
              style={styles.textButton}
              labelStyle={styles.textButtonLabel}
            >
              Already have an account? Sign In
            </Button>
          </Link>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1e3c72',
  },
  backgroundGradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  decorativeShapes: {
    position: 'absolute',
    width: '100%',
    height: '100%',
  },
  circle: {
    position: 'absolute',
    borderRadius: 1000,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  circle1: {
    width: 200,
    height: 200,
    top: -100,
    right: -50,
  },
  circle2: {
    width: 150,
    height: 150,
    bottom: 100,
    left: -75,
  },
  circle3: {
    width: 100,
    height: 100,
    top: height * 0.3,
    right: 20,
  },
  shape: {
    position: 'absolute',
  },
  shape1: {
    width: 300,
    height: 300,
    borderRadius: 150,
    top: -150,
    left: -100,
    transform: [{ rotate: '45deg' }],
  },
  shape2: {
    width: 250,
    height: 250,
    borderRadius: 125,
    bottom: -125,
    right: -50,
    transform: [{ rotate: '-30deg' }],
  },
  contentOverlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
    zIndex: 10,
  },
  titleSection: {
    alignItems: 'center',
    marginBottom: 60,
  },
  title: {
    color: '#fff',
    marginBottom: 8,
    fontSize: 28,
    fontWeight: '300',
    textShadow: '0px 2px 4px rgba(0,0,0,0.3)',
  },
  brandName: {
    color: '#fff',
    fontWeight: 'bold',
    marginBottom: 16,
    fontSize: 48,
    letterSpacing: 2,
    textShadow: '0px 4px 8px rgba(0,0,0,0.3)',
  },
  subtitle: {
    color: '#f8f9fa',
    textAlign: 'center',
    fontSize: 18,
    lineHeight: 24,
    fontWeight: '400',
    textShadow: '0px 1px 3px rgba(0,0,0,0.3)',
  },
  buttonContainer: {
    width: '100%',
    maxWidth: 320,
    gap: 20,
  },
  button: {
    width: '100%',
    borderRadius: 25,
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 8,
    },
    shadowOpacity: 0.2,
    shadowRadius: 20,
    elevation: 8,
  },
  buttonContent: {
    height: 60,
  },
  buttonLabel: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1e3c72',
  },
  textButton: {
    marginTop: 0,
    paddingVertical: 8,
  },
  textButtonLabel: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
    textShadow: '0px 1px 2px rgba(0,0,0,0.3)',
  },
}); 