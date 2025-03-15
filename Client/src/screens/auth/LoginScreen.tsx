import React, { useState } from 'react';
import { View, StyleSheet, Image, TouchableOpacity } from 'react-native';
import { TextInput, Button, Title, Text, HelperText } from 'react-native-paper';
import { useAuth } from '../../context/AuthContext';

const LoginScreen = ({ navigation }) => {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async () => {
    if (!email || !password) {
      setError('Please enter both email and password');
      return;
    }

    setLoading(true);
    setError('');

    try {
      await login(email, password);
    } catch (err) {
      setError('Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.logoContainer}>
        <Image
          source={require('../../assets/logo.png')}
          style={styles.logo}
          resizeMode="contain"
        />
        <Title style={styles.title}>Alzheimer's AI Companion</Title>
      </View>

      <View style={styles.formContainer}>
        <TextInput
          label="Email"
          value={email}
          onChangeText={setEmail}
          mode="outlined"
          style={styles.input}
          keyboardType="email-address"
          autoCapitalize="none"
        />

        <TextInput
          label="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          mode="outlined"
          style={styles.input}
        />

        {error ? <HelperText type="error">{error}</HelperText> : null}

        <Button
          mode="contained"
          onPress={handleLogin}
          style={styles.button}
          loading={loading}
          disabled={loading}
        >
          Log In
        </Button>

        <Button
          mode="outlined"
          onPress={() => {/* Implement Google login */}}
          style={styles.googleButton}
          icon="google"
        >
          Sign in with Google
        </Button>

        <View style={styles.signupContainer}>
          <Text>Don't have an account? </Text>
          <TouchableOpacity onPress={() => navigation.navigate('Signup')}>
            <Text style={styles.signupText}>Sign Up</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  logoContainer: {
    alignItems: 'center',
    marginTop: 60,
    marginBottom: 40,
  },
  logo: {
    width: 120,
    height: 120,
  },
  title: {
    fontSize: 24,
    marginTop: 10,
    textAlign: 'center',
  },
  formContainer: {
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 20,
    elevation: 2,
  },
  input: {
    marginBottom: 15,
  },
  button: {
    marginTop: 10,
    marginBottom: 15,
    paddingVertical: 6,
  },
  googleButton: {
    marginBottom: 15,
  },
  signupContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 10,
  },
  signupText: {
    color: '#6200ee',
    fontWeight: 'bold',
  },
});

export default LoginScreen;
