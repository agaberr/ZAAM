import React from 'react';
import { StyleSheet, View, Dimensions } from 'react-native';
import { Button, Text } from 'react-native-paper';
import { router } from 'expo-router';
import { Video, ResizeMode } from 'expo-av';
import { LinearGradient } from 'expo-linear-gradient';

const { width, height } = Dimensions.get('window');

export default function WelcomeScreen() {
  const handleEmailSignUp = () => {
    router.push('/auth/signup');
  };

  const handleSignIn = () => {
    router.push('/auth/signin');
  };

  const handleGoogleAuth = () => {
    // TODO: Implement Google authentication
    console.log('Google auth pressed');
  };

  return (
    <View style={styles.container}>
      <Video
        source={require('./assets/detroit_video.mp4')}
        style={styles.backgroundVideo}
        resizeMode={ResizeMode.COVER}
        shouldPlay
        isLooping
        isMuted
      />
      <LinearGradient
        colors={['rgba(0,0,0,0.7)', 'rgba(0,0,0,0.5)', 'rgba(0,0,0,0.7)']}
        style={styles.gradient}
      />
      <View style={styles.content}>
        <Text variant="displaySmall" style={styles.title}>
          Welcome to
        </Text>
        <Text variant="displayLarge" style={styles.brandName}>
          ZAAM
        </Text>
        <Text variant="titleMedium" style={styles.subtitle}>
          Your AI Companion for Alzheimer's Care
        </Text>
        <View style={styles.buttonContainer}>
          <Button
            mode="contained"
            onPress={handleEmailSignUp}
            style={styles.button}
            contentStyle={styles.buttonContent}
            labelStyle={styles.buttonLabel}
          >
            Create Account
          </Button>
          <Button
            mode="outlined"
            onPress={handleGoogleAuth}
            style={[styles.button, styles.googleButton]}
            contentStyle={styles.buttonContent}
            labelStyle={styles.googleButtonLabel}
            icon="google"
          >
            Continue with Google
          </Button>
          <Button
            mode="text"
            onPress={handleSignIn}
            style={styles.textButton}
            labelStyle={styles.textButtonLabel}
          >
            Already have an account? Sign In
          </Button>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  backgroundVideo: {
    position: 'absolute',
    top: 0,
    left: 0,
    bottom: 0,
    right: 0,
    width,
    height,
  },
  gradient: {
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
    paddingHorizontal: 24,
  },
  title: {
    color: '#fff',
    marginBottom: 8,
  },
  brandName: {
    color: '#fff',
    fontWeight: 'bold',
    marginBottom: 16,
  },
  subtitle: {
    color: '#fff',
    textAlign: 'center',
    marginBottom: 48,
    opacity: 0.9,
  },
  buttonContainer: {
    width: '100%',
    maxWidth: 320,
    gap: 16,
  },
  button: {
    width: '100%',
    borderRadius: 12,
  },
  buttonContent: {
    height: 56,
  },
  buttonLabel: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  googleButton: {
    backgroundColor: '#fff',
    borderWidth: 0,
  },
  googleButtonLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#000',
  },
  textButton: {
    marginTop: 8,
  },
  textButtonLabel: {
    color: '#fff',
    fontSize: 14,
  },
}); 