import React, { useState } from 'react';
import { StyleSheet, View, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { Text, TextInput, Button, IconButton } from 'react-native-paper';
import { router } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../context/AuthContext';

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
      setError('Invalid username or password. Try using "ahmed" for both.');
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
      <LinearGradient
        colors={['#4285F4', '#34A853']}
        style={styles.background}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
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
              
              <Button
                mode="text"
                onPress={() => console.log('Forgot password')}
                style={styles.forgotButton}
                labelStyle={styles.forgotButtonLabel}
              >
                Forgot Password?
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
  },
  background: {
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
  },
  title: {
    color: 'white',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    color: 'rgba(255,255,255,0.8)',
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
    borderRadius: 12,
  },
  buttonContent: {
    height: 56,
  },
  buttonLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#4285F4',
  },
  forgotButton: {
    marginTop: 16,
  },
  forgotButtonLabel: {
    color: 'white',
  },
}); 