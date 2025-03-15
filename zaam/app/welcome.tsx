import React from 'react';
import { StyleSheet, View, Image } from 'react-native';
import { Button, Text, useTheme } from 'react-native-paper';
import { router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

export default function WelcomeScreen() {
  const theme = useTheme();

  const handleEmailSignUp = () => {
    router.push('/auth/signup');
  };

  const handleEmailSignIn = () => {
    router.push('/auth/signin');
  };

  const handleGoogleAuth = async () => {
    // TODO: Implement Google Authentication
    console.log('Google Auth pressed');
  };

  return (
    <View style={styles.container}>
      <StatusBar style="auto" />
      <View style={styles.logoContainer}>
        <Image
          source={require('./assets/icon.png')}
          style={styles.logo}
          resizeMode="contain"
        />
        <Text variant="headlineLarge" style={styles.title}>
          ZAAM
        </Text>
        <Text variant="titleMedium" style={styles.subtitle}>
          Your AI Companion for Daily Support
        </Text>
      </View>

      <View style={styles.buttonContainer}>
        <Button
          mode="contained"
          onPress={handleEmailSignUp}
          style={styles.button}
          contentStyle={styles.buttonContent}
          labelStyle={styles.buttonLabel}
        >
          Sign Up with Email
        </Button>

        <Button
          mode="contained-tonal"
          onPress={handleGoogleAuth}
          style={styles.button}
          contentStyle={styles.buttonContent}
          labelStyle={styles.buttonLabel}
          icon="google"
        >
          Continue with Google
        </Button>

        <Button
          mode="outlined"
          onPress={handleEmailSignIn}
          style={styles.button}
          contentStyle={styles.buttonContent}
          labelStyle={styles.buttonLabel}
        >
          Already have an account? Sign In
        </Button>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 20,
    justifyContent: 'space-between',
  },
  logoContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 60,
  },
  logo: {
    width: 150,
    height: 150,
    marginBottom: 20,
  },
  title: {
    marginBottom: 8,
    textAlign: 'center',
    fontWeight: 'bold',
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.7,
  },
  buttonContainer: {
    width: '100%',
    paddingBottom: 20,
  },
  button: {
    marginVertical: 8,
    borderRadius: 8,
  },
  buttonContent: {
    paddingVertical: 8,
  },
  buttonLabel: {
    fontSize: 16,
  },
}); 