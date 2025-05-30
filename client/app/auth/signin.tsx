import React, { useState } from 'react';
import { StyleSheet, View, KeyboardAvoidingView, Platform, ScrollView, Dimensions } from 'react-native';
import { Text, TextInput, Button, IconButton } from 'react-native-paper';
import { router } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../context/AuthContext';

const { width, height } = Dimensions.get('window');

export default function SignInScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const { signIn } = useAuth();

  const handleSignIn = async () => {
    if (!email || !password) {
      setError('Please enter both email and password');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      await signIn(email, password);
      // No need to navigate here - the auth context will handle it
    } catch (err) {
      setError('Invalid username or password.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const goBack = () => {
    router.back();
  };

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
      
      <IconButton
        icon="arrow-left"
        size={24}
        onPress={goBack}
        style={styles.backButton}
        iconColor="white"
      />
      
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardAvoidingView}
      >
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <View style={styles.content}>
            <Text variant="headlineMedium" style={styles.title}>Sign In</Text>
            <Text variant="bodyLarge" style={styles.subtitle}>
              Welcome back! Please sign in to continue.
            </Text>
            
            {error ? <Text style={styles.errorText}>{error}</Text> : null}
            
            <View style={styles.form}>
              <TextInput
                label="Email"
                value={email}
                onChangeText={setEmail}
                mode="flat"
                style={styles.input}
                keyboardType="email-address"
                autoCapitalize="none"
                underlineColor="rgba(255,255,255,0.3)"
                activeUnderlineColor="white"
                textColor="white"
                theme={{ 
                  colors: { 
                    onSurfaceVariant: 'white',
                    text: 'white',
                    placeholder: 'white' 
                  } 
                }}
              />
              
              <TextInput
                label="Password"
                value={password}
                onChangeText={setPassword}
                mode="flat"
                style={styles.input}
                secureTextEntry={!showPassword}
                right={
                  <TextInput.Icon
                    icon={showPassword ? 'eye-off' : 'eye'}
                    onPress={() => setShowPassword(!showPassword)}
                    color="white"
                  />
                }
                underlineColor="rgba(255,255,255,0.3)"
                activeUnderlineColor="white"
                textColor="white"
                theme={{ 
                  colors: { 
                    onSurfaceVariant: 'white',
                    text: 'white',
                    placeholder: 'white' 
                  } 
                }}
              />
              
              <Button
                mode="contained"
                onPress={handleSignIn}
                style={styles.button}
                contentStyle={styles.buttonContent}
                labelStyle={styles.buttonLabel}
                loading={loading}
                disabled={loading}
              >
                Sign In
              </Button>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
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
  backButton: {
    position: 'absolute',
    top: 40,
    left: 16,
    zIndex: 10,
  },
  keyboardAvoidingView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
  },
  content: {
    padding: 24,
    width: '100%',
    maxWidth: 400,
    alignSelf: 'center',
    zIndex: 5,
  },
  title: {
    color: 'white',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    color: 'rgba(255,255,255,0.9)',
    marginBottom: 32,
  },
  errorText: {
    color: '#FF3B30',
    backgroundColor: 'rgba(255, 59, 48, 0.1)',
    padding: 10,
    borderRadius: 5,
    marginBottom: 16,
  },
  form: {
    width: '100%',
  },
  input: {
    marginBottom: 24,
    backgroundColor: 'transparent',
    height: 60,
    paddingBottom: 8,
  },
  button: {
    marginTop: 16,
    backgroundColor: 'white',
    borderRadius: 25,
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
}); 